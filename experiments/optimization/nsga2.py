"""NSGA-II: Non-dominated Sorting Genetic Algorithm II.

Budget-matched counterpart to MOPSO for the algorithm comparison
(reviewer request: substantiate the MOPSO advantage with multi-run
significance tests and complementary indicators):
  - population size P, G generations -> P * (G + 1) evaluations,
    identical to MOPSO with P particles and G iterations
  - fast non-dominated sorting + crowding distance (Deb et al. 2002)
  - binary tournament selection on (rank, crowding distance)
  - SBX crossover and polynomial mutation (bounded variants)
  - same bounds, discretization and constraint handling as MOPSO;
    the constraint alpha_p > alpha_d + delta is penalized inside the
    objective function (see optimization/objectives.py)
"""

import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MOPSO_SWARM_SIZE, MOPSO_ITERATIONS, RANDOM_SEED
from optimization.objectives import (
    BOUNDS_LOWER, BOUNDS_UPPER, N_DIMS, N_OBJECTIVES,
    discretize_particle,
)
from optimization.mopso import dominates, crowding_distance


def fast_nondominated_sort(objectives):
    """Fast non-dominated sorting (Deb et al. 2002).

    Args:
        objectives: (n, m) array of objective vectors (all minimized)
    Returns:
        fronts: list of lists of indices, fronts[0] is the Pareto front
        ranks: (n,) array, ranks[i] = front index of solution i
    """
    n = len(objectives)
    dominated_by = [[] for _ in range(n)]  # solutions that i dominates
    n_dominators = np.zeros(n, dtype=int)  # count of solutions dominating i
    fronts = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(objectives[i], objectives[j]):
                dominated_by[i].append(j)
            elif dominates(objectives[j], objectives[i]):
                n_dominators[i] += 1
        if n_dominators[i] == 0:
            fronts[0].append(i)

    ranks = np.zeros(n, dtype=int)
    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in dominated_by[i]:
                n_dominators[j] -= 1
                if n_dominators[j] == 0:
                    ranks[j] = k + 1
                    next_front.append(j)
        k += 1
        fronts.append(next_front)
    fronts.pop()  # last front is empty
    return fronts, ranks


class NSGA2:
    """NSGA-II with SBX crossover and polynomial mutation."""

    def __init__(self, objective_fn, pop_size=MOPSO_SWARM_SIZE,
                 n_generations=MOPSO_ITERATIONS, seed=RANDOM_SEED,
                 eta_crossover=15.0, eta_mutation=20.0,
                 p_crossover=0.9, p_mutation=None):
        self.objective_fn = objective_fn
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.rng = np.random.RandomState(seed)

        self.eta_c = eta_crossover
        self.eta_m = eta_mutation
        self.p_crossover = p_crossover
        self.p_mutation = (1.0 / N_DIMS) if p_mutation is None else p_mutation

        # Initialize population (same LHS scheme as MOPSO for fairness)
        self.positions = self._init_positions()
        for i in range(pop_size):
            self.positions[i] = discretize_particle(self.positions[i])
        self.objectives = np.array(
            [self.objective_fn(p) for p in self.positions])

        self._fronts, self._ranks = fast_nondominated_sort(self.objectives)
        self._crowding = self._compute_crowding(self._fronts,
                                                self.objectives)

    def _init_positions(self):
        """Latin Hypercube Sampling for better initial coverage."""
        positions = np.zeros((self.pop_size, N_DIMS))
        for d in range(N_DIMS):
            intervals = np.linspace(BOUNDS_LOWER[d], BOUNDS_UPPER[d],
                                    self.pop_size + 1)
            for i in range(self.pop_size):
                positions[i, d] = self.rng.uniform(intervals[i],
                                                   intervals[i+1])
            self.rng.shuffle(positions[:, d])
        return positions

    @staticmethod
    def _compute_crowding(fronts, objectives):
        """Per-individual crowding distance, computed within each front."""
        crowding = np.zeros(len(objectives))
        for front in fronts:
            cd = crowding_distance(objectives[np.array(front)])
            for idx, c in zip(front, cd):
                crowding[idx] = c
        return crowding

    def _tournament(self):
        """Binary tournament on (rank, crowding distance)."""
        i, j = self.rng.randint(self.pop_size, size=2)
        if self._ranks[i] < self._ranks[j]:
            return i
        if self._ranks[j] < self._ranks[i]:
            return j
        return i if self._crowding[i] > self._crowding[j] else j

    def _sbx_crossover(self, parent1, parent2):
        """Simulated Binary Crossover (bounded)."""
        child1, child2 = parent1.copy(), parent2.copy()
        if self.rng.rand() > self.p_crossover:
            return child1, child2

        for d in range(N_DIMS):
            if self.rng.rand() > 0.5:
                continue
            if abs(parent1[d] - parent2[d]) < 1e-14:
                continue
            y1 = min(parent1[d], parent2[d])
            y2 = max(parent1[d], parent2[d])
            lo, hi = BOUNDS_LOWER[d], BOUNDS_UPPER[d]
            span = y2 - y1
            u = self.rng.rand()

            beta = 1.0 + 2.0 * (y1 - lo) / span
            alpha = 2.0 - beta ** (-(self.eta_c + 1.0))
            if u <= 1.0 / alpha:
                betaq = (u * alpha) ** (1.0 / (self.eta_c + 1.0))
            else:
                betaq = (1.0 / (2.0 - u * alpha)) ** (1.0 / (self.eta_c + 1.0))
            c1 = 0.5 * ((y1 + y2) - betaq * span)

            beta = 1.0 + 2.0 * (hi - y2) / span
            alpha = 2.0 - beta ** (-(self.eta_c + 1.0))
            if u <= 1.0 / alpha:
                betaq = (u * alpha) ** (1.0 / (self.eta_c + 1.0))
            else:
                betaq = (1.0 / (2.0 - u * alpha)) ** (1.0 / (self.eta_c + 1.0))
            c2 = 0.5 * ((y1 + y2) + betaq * span)

            c1 = np.clip(c1, lo, hi)
            c2 = np.clip(c2, lo, hi)
            if self.rng.rand() < 0.5:
                c1, c2 = c2, c1
            child1[d], child2[d] = c1, c2
        return child1, child2

    def _polynomial_mutation(self, x):
        """Polynomial mutation (bounded)."""
        y = x.copy()
        for d in range(N_DIMS):
            if self.rng.rand() > self.p_mutation:
                continue
            lo, hi = BOUNDS_LOWER[d], BOUNDS_UPPER[d]
            if hi - lo < 1e-14:
                continue
            delta1 = (y[d] - lo) / (hi - lo)
            delta2 = (hi - y[d]) / (hi - lo)
            u = self.rng.rand()
            mut_pow = 1.0 / (self.eta_m + 1.0)
            if u < 0.5:
                val = 2.0 * u + (1.0 - 2.0 * u) * (1.0 - delta1) ** (self.eta_m + 1.0)
                deltaq = val ** mut_pow - 1.0
            else:
                val = (2.0 * (1.0 - u)
                       + 2.0 * (u - 0.5) * (1.0 - delta2) ** (self.eta_m + 1.0))
                deltaq = 1.0 - val ** mut_pow
            y[d] = np.clip(y[d] + deltaq * (hi - lo), lo, hi)
        return y

    def _make_offspring(self):
        """Generate pop_size offspring via tournament + SBX + mutation."""
        offspring = []
        while len(offspring) < self.pop_size:
            p1 = self.positions[self._tournament()]
            p2 = self.positions[self._tournament()]
            c1, c2 = self._sbx_crossover(p1, p2)
            for child in (c1, c2):
                if len(offspring) >= self.pop_size:
                    break
                child = self._polynomial_mutation(child)
                child = discretize_particle(child)
                offspring.append(child)
        return np.array(offspring)

    def optimize(self, verbose=True):
        """Run NSGA-II optimization loop.

        Returns:
            pareto_positions: list of parameter vectors on Pareto front
            pareto_objectives: list of objective vectors on Pareto front
            history: dict with generation-level metrics
        """
        history = {"n_front": []}

        for gen in range(self.n_generations):
            offspring = self._make_offspring()
            offspring_obj = np.array(
                [self.objective_fn(c) for c in offspring])

            # Environmental selection on combined parent + offspring pool
            combined_pos = np.vstack([self.positions, offspring])
            combined_obj = np.vstack([self.objectives, offspring_obj])
            fronts, _ = fast_nondominated_sort(combined_obj)

            next_idx = []
            for front in fronts:
                if len(next_idx) + len(front) <= self.pop_size:
                    next_idx.extend(front)
                else:
                    # Truncate last front by crowding distance
                    cd = crowding_distance(combined_obj[np.array(front)])
                    order = np.argsort(-cd)
                    n_needed = self.pop_size - len(next_idx)
                    next_idx.extend([front[k] for k in order[:n_needed]])
                if len(next_idx) >= self.pop_size:
                    break

            next_idx = np.array(next_idx)
            self.positions = combined_pos[next_idx]
            self.objectives = combined_obj[next_idx]
            self._fronts, self._ranks = fast_nondominated_sort(
                self.objectives)
            self._crowding = self._compute_crowding(self._fronts,
                                                    self.objectives)

            history["n_front"].append(len(self._fronts[0]))

            if verbose and (gen + 1) % 10 == 0:
                front_obj = self.objectives[np.array(self._fronts[0])]
                print(f"Gen {gen+1}/{self.n_generations}  "
                      f"front={len(self._fronts[0])}  "
                      f"min_f1={front_obj[:, 0].min():.4f}  "
                      f"min_f2={front_obj[:, 1].min():.4f}")

        front_idx = np.array(self._fronts[0])
        return (
            [self.positions[i].copy() for i in front_idx],
            [self.objectives[i].copy() for i in front_idx],
            history,
        )

    def get_balanced_solution(self):
        """Select the Pareto solution closest to the ideal point."""
        front_idx = np.array(self._fronts[0])
        objs = self.objectives[front_idx]
        obj_min = objs.min(axis=0)
        obj_max = objs.max(axis=0)
        obj_range = obj_max - obj_min
        obj_range[obj_range < 1e-10] = 1.0
        normalized = (objs - obj_min) / obj_range
        distances = np.sqrt((normalized ** 2).sum(axis=1))
        best = front_idx[np.argmin(distances)]
        return self.positions[best], self.objectives[best]

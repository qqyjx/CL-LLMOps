"""MOPSO: Multi-Objective Particle Swarm Optimization.

Implementation following paper Section 4.5:
  - 50 particles, 100 iterations
  - External archive with crowding distance for leader selection
  - Linear inertia decay from 0.9 to 0.4
  - Constraint handling via penalty in objective function
  - Archive size limited to 30 (pruned by crowding distance)
"""

import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    MOPSO_SWARM_SIZE, MOPSO_ITERATIONS, MOPSO_ARCHIVE_SIZE,
    MOPSO_INERTIA, MOPSO_C1, MOPSO_C2, RANDOM_SEED,
)
from optimization.objectives import (
    BOUNDS_LOWER, BOUNDS_UPPER, N_DIMS, N_OBJECTIVES,
    discretize_particle,
)


def dominates(a, b):
    """Check if solution a dominates solution b (all objectives minimized)."""
    return np.all(a <= b) and np.any(a < b)


def crowding_distance(objectives):
    """Compute crowding distance for a set of objective vectors."""
    n = len(objectives)
    if n <= 2:
        return np.full(n, np.inf)

    distances = np.zeros(n)
    for m in range(objectives.shape[1]):
        sorted_idx = np.argsort(objectives[:, m])
        distances[sorted_idx[0]] = np.inf
        distances[sorted_idx[-1]] = np.inf

        obj_range = objectives[sorted_idx[-1], m] - objectives[sorted_idx[0], m]
        if obj_range < 1e-10:
            continue

        for i in range(1, n - 1):
            distances[sorted_idx[i]] += (
                objectives[sorted_idx[i+1], m] - objectives[sorted_idx[i-1], m]
            ) / obj_range

    return distances


class MOPSO:
    """Multi-Objective Particle Swarm Optimization with external archive."""

    def __init__(self, objective_fn, n_particles=MOPSO_SWARM_SIZE,
                 n_iterations=MOPSO_ITERATIONS, archive_size=MOPSO_ARCHIVE_SIZE,
                 seed=RANDOM_SEED):
        self.objective_fn = objective_fn
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.archive_size = archive_size
        self.rng = np.random.RandomState(seed)

        self.w_max, self.w_min = MOPSO_INERTIA
        self.c1 = MOPSO_C1
        self.c2 = MOPSO_C2

        # Initialize swarm
        self.positions = self._init_positions()
        self.velocities = np.zeros((n_particles, N_DIMS))
        self.objectives = np.array([self.objective_fn(p) for p in self.positions])

        self.pbest_pos = self.positions.copy()
        self.pbest_obj = self.objectives.copy()

        # External archive (Pareto front)
        self.archive_pos = []
        self.archive_obj = []
        self._update_archive(self.positions, self.objectives)

    def _init_positions(self):
        """Latin Hypercube Sampling for better initial coverage."""
        positions = np.zeros((self.n_particles, N_DIMS))
        for d in range(N_DIMS):
            intervals = np.linspace(BOUNDS_LOWER[d], BOUNDS_UPPER[d],
                                    self.n_particles + 1)
            for i in range(self.n_particles):
                positions[i, d] = self.rng.uniform(intervals[i], intervals[i+1])
            self.rng.shuffle(positions[:, d])
        return positions

    def _select_leader(self):
        """Select a global guide from archive using crowding distance roulette."""
        if len(self.archive_obj) == 0:
            return self.positions[self.rng.randint(self.n_particles)]

        objs = np.array(self.archive_obj)
        cd = crowding_distance(objs)
        # Replace inf with large finite value for roulette
        cd = np.where(np.isinf(cd), 1e6, cd)
        probs = cd / cd.sum()
        idx = self.rng.choice(len(self.archive_pos), p=probs)
        return self.archive_pos[idx]

    def _update_archive(self, new_positions, new_objectives):
        """Add non-dominated solutions to archive, prune if oversized."""
        candidates_pos = list(self.archive_pos) + list(new_positions)
        candidates_obj = list(self.archive_obj) + list(new_objectives)

        # Find non-dominated set
        n = len(candidates_pos)
        is_dominated = [False] * n
        for i in range(n):
            if is_dominated[i]:
                continue
            for j in range(n):
                if i == j or is_dominated[j]:
                    continue
                if dominates(np.array(candidates_obj[j]),
                             np.array(candidates_obj[i])):
                    is_dominated[i] = True
                    break

        self.archive_pos = [candidates_pos[i] for i in range(n)
                            if not is_dominated[i]]
        self.archive_obj = [candidates_obj[i] for i in range(n)
                            if not is_dominated[i]]

        # Prune by crowding distance if over capacity
        if len(self.archive_pos) > self.archive_size:
            objs = np.array(self.archive_obj)
            cd = crowding_distance(objs)
            # Keep the most diverse solutions
            keep_idx = np.argsort(-cd)[:self.archive_size]
            self.archive_pos = [self.archive_pos[i] for i in keep_idx]
            self.archive_obj = [self.archive_obj[i] for i in keep_idx]

    def optimize(self, verbose=True):
        """Run MOPSO optimization loop.

        Returns:
            pareto_positions: list of parameter vectors on Pareto front
            pareto_objectives: list of objective vectors on Pareto front
            history: dict with iteration-level metrics
        """
        history = {"n_archive": [], "hypervolume_approx": []}

        for it in range(self.n_iterations):
            # Linear inertia decay
            w = self.w_max - (self.w_max - self.w_min) * it / self.n_iterations

            for i in range(self.n_particles):
                leader = self._select_leader()
                r1 = self.rng.rand(N_DIMS)
                r2 = self.rng.rand(N_DIMS)

                # Velocity update
                self.velocities[i] = (
                    w * self.velocities[i]
                    + self.c1 * r1 * (self.pbest_pos[i] - self.positions[i])
                    + self.c2 * r2 * (leader - self.positions[i])
                )

                # Position update
                self.positions[i] += self.velocities[i]

                # Boundary handling: reflect
                for d in range(N_DIMS):
                    if self.positions[i, d] < BOUNDS_LOWER[d]:
                        self.positions[i, d] = BOUNDS_LOWER[d]
                        self.velocities[i, d] *= -0.5
                    elif self.positions[i, d] > BOUNDS_UPPER[d]:
                        self.positions[i, d] = BOUNDS_UPPER[d]
                        self.velocities[i, d] *= -0.5

                # Discretize
                self.positions[i] = discretize_particle(self.positions[i])

                # Evaluate
                self.objectives[i] = self.objective_fn(self.positions[i])

                # Update personal best
                if dominates(self.objectives[i], self.pbest_obj[i]):
                    self.pbest_pos[i] = self.positions[i].copy()
                    self.pbest_obj[i] = self.objectives[i].copy()

            # Update archive
            self._update_archive(self.positions, self.objectives)

            # Record history
            history["n_archive"].append(len(self.archive_pos))

            if verbose and (it + 1) % 10 == 0:
                objs = np.array(self.archive_obj)
                print(f"Iter {it+1}/{self.n_iterations}  "
                      f"archive={len(self.archive_pos)}  "
                      f"best_f1={-objs[:, 0].min():.4f}  "
                      f"min_halluc={objs[:, 1].min():.4f}")

        return (
            [p.copy() for p in self.archive_pos],
            [o.copy() for o in self.archive_obj],
            history,
        )

    def get_balanced_solution(self):
        """Select the Pareto solution closest to the ideal point."""
        if not self.archive_obj:
            return None, None

        objs = np.array(self.archive_obj)
        # Normalize each objective to [0, 1]
        obj_min = objs.min(axis=0)
        obj_max = objs.max(axis=0)
        obj_range = obj_max - obj_min
        obj_range[obj_range < 1e-10] = 1.0
        normalized = (objs - obj_min) / obj_range

        # Find closest to origin (ideal point)
        distances = np.sqrt((normalized ** 2).sum(axis=1))
        best_idx = np.argmin(distances)
        return self.archive_pos[best_idx], self.archive_obj[best_idx]

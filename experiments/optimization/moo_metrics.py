"""Multi-objective performance indicators for the MOPSO vs NSGA-II study.

Indicators (all computed on minimization fronts):
  - hypervolume: exact, via coordinate-compressed grid (inclusion of
    dominated cells); intended for 2-3 objectives and moderate fronts
  - igd: Inverted Generational Distance against a reference front
    (in this project: merged non-dominated set across all runs of
    both algorithms)
  - generalized_spread: Delta* indicator of Zhou et al. 2006, valid
    for >2 objectives (uses reference-front extreme points)

All functions are pure numpy. Fronts are (n, m) arrays. Callers are
expected to normalize objectives to a common scale first (see
normalize_front) so the hypervolume reference point can be a constant
such as (1.1, ..., 1.1).
"""

from functools import reduce

import numpy as np


def nondominated_front(points):
    """Extract the non-dominated subset of a set of objective vectors.

    Args:
        points: (n, m) array, all objectives minimized
    Returns:
        (k, m) array of unique non-dominated points
    """
    points = np.unique(np.asarray(points, dtype=float), axis=0)
    n = len(points)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated = (np.all(points <= points[i], axis=1)
                     & np.any(points < points[i], axis=1))
        if dominated.any():
            keep[i] = False
            continue
        dominates_i = (np.all(points[i] <= points, axis=1)
                       & np.any(points[i] < points, axis=1))
        keep[dominates_i] = False
    return points[keep]


def normalize_front(points, ideal, nadir):
    """Map objectives to [0, 1] using ideal/nadir of the reference front."""
    ideal = np.asarray(ideal, dtype=float)
    nadir = np.asarray(nadir, dtype=float)
    span = nadir - ideal
    span[span < 1e-12] = 1.0
    return (np.asarray(points, dtype=float) - ideal) / span


def hypervolume(points, ref_point):
    """Exact hypervolume dominated by `points` w.r.t. `ref_point`.

    Uses coordinate compression: the union of boxes [p, ref] is
    decomposed into grid cells; a cell is dominated iff some point is
    <= its lower corner. Exact for any m, practical for m <= 3 and
    fronts up to a few hundred points.

    Args:
        points: (n, m) array, all objectives minimized
        ref_point: (m,) reference (worst) point
    Returns:
        float hypervolume (0.0 if no point strictly dominates ref_point)
    """
    points = np.asarray(points, dtype=float)
    ref_point = np.asarray(ref_point, dtype=float)
    if points.ndim != 2 or points.shape[1] != len(ref_point):
        raise ValueError("points must be (n, m) matching ref_point")

    points = points[np.all(points < ref_point, axis=1)]
    if len(points) == 0:
        return 0.0
    points = nondominated_front(points)

    m = len(ref_point)
    coords = [np.unique(np.concatenate([points[:, d], [ref_point[d]]]))
              for d in range(m)]
    widths = [np.diff(c) for c in coords]
    lowers = [c[:-1] for c in coords]

    covered = np.zeros([len(w) for w in widths], dtype=bool)
    for p in points:
        masks = [p[d] <= lowers[d] for d in range(m)]
        covered |= reduce(np.multiply.outer, masks).astype(bool)

    cell_vol = reduce(np.multiply.outer, widths)
    return float((covered * cell_vol).sum())


def igd(front, reference_front):
    """Inverted Generational Distance (mean, Euclidean).

    Average distance from each reference-front point to its nearest
    point in the approximation front. Lower is better.
    """
    front = np.asarray(front, dtype=float)
    reference_front = np.asarray(reference_front, dtype=float)
    if len(front) == 0:
        return float("inf")
    dists = np.linalg.norm(
        reference_front[:, None, :] - front[None, :, :], axis=2)
    return float(dists.min(axis=1).mean())


def generalized_spread(front, reference_front):
    """Generalized spread Delta* (Zhou et al. 2006) for m >= 2 objectives.

    Delta* = (sum_m d(e_m, S) + sum_i |d_i - d_mean|)
             / (sum_m d(e_m, S) + n * d_mean)
    where e_m are the extreme points of the reference front (maximizers
    of each objective) and d_i is the nearest-neighbor distance within
    the approximation front S. Lower is better (0 = ideally uniform
    front spanning the extremes).
    """
    front = np.asarray(front, dtype=float)
    reference_front = np.asarray(reference_front, dtype=float)
    if len(front) < 2:
        return 1.0

    m = front.shape[1]
    extremes = np.array([reference_front[np.argmax(reference_front[:, d])]
                         for d in range(m)])
    d_ext = np.linalg.norm(
        extremes[:, None, :] - front[None, :, :], axis=2).min(axis=1)

    pairwise = np.linalg.norm(
        front[:, None, :] - front[None, :, :], axis=2)
    np.fill_diagonal(pairwise, np.inf)
    d_nn = pairwise.min(axis=1)
    d_mean = d_nn.mean()

    denom = d_ext.sum() + len(front) * d_mean
    if denom < 1e-12:
        return 0.0
    return float((d_ext.sum() + np.abs(d_nn - d_mean).sum()) / denom)


if __name__ == "__main__":
    # ── Hand-checkable self-tests ──
    ref = np.array([1.0, 1.0, 1.0])

    # 1. Single point at origin dominates the whole unit cube
    assert abs(hypervolume(np.array([[0.0, 0.0, 0.0]]), ref) - 1.0) < 1e-12

    # 2. Single interior point: (1-0.5)^3 = 0.125
    assert abs(hypervolume(np.array([[0.5, 0.5, 0.5]]), ref) - 0.125) < 1e-12

    # 3. Two overlapping boxes, inclusion-exclusion by hand:
    #    v1 = 0.75*0.75*0.5 = 0.28125, v2 = 0.5*0.5*0.75 = 0.1875,
    #    overlap = 0.5*0.5*0.5 = 0.125 -> union = 0.34375
    two = np.array([[0.25, 0.25, 0.5], [0.5, 0.5, 0.25]])
    assert abs(hypervolume(two, ref) - 0.34375) < 1e-12

    # 4. Dominated point contributes nothing
    three = np.vstack([two, [[0.6, 0.6, 0.6]]])
    assert abs(hypervolume(three, ref) - 0.34375) < 1e-12

    # 5. Point outside the reference box is ignored
    assert hypervolume(np.array([[1.5, 0.2, 0.2]]), ref) == 0.0

    # 6. IGD: identical fronts -> 0; shifted by 0.1 in one axis -> 0.1
    f = np.array([[0.0, 1.0, 0.5], [1.0, 0.0, 0.5]])
    assert igd(f, f) == 0.0
    shifted = f + np.array([0.1, 0.0, 0.0])
    assert abs(igd(shifted, f) - 0.1) < 1e-12

    # 7. Non-dominated filtering
    pts = np.array([[0.1, 0.9, 0.5], [0.9, 0.1, 0.5], [0.5, 0.5, 0.6],
                    [0.6, 0.95, 0.55]])  # last dominated by first
    nd = nondominated_front(pts)
    assert len(nd) == 3

    # 8. Spread: perfectly uniform 2-obj front spanning the extremes -> 0
    line = np.array([[x, 1.0 - x] for x in np.linspace(0, 1, 11)])
    assert generalized_spread(line, line) < 1e-12
    # Non-uniform front has strictly larger spread
    clumped = np.array([[0.0, 1.0], [0.05, 0.95], [0.1, 0.9], [1.0, 0.0]])
    assert generalized_spread(clumped, line) > 0.3

    # 9. Normalization maps ideal->0, nadir->1
    norm = normalize_front(np.array([[0.5, 5.0, 0.0]]),
                           ideal=[0.0, 0.0, 0.0], nadir=[1.0, 10.0, 1.0])
    assert np.allclose(norm, [[0.5, 0.5, 0.0]])

    print("moo_metrics self-tests passed.")

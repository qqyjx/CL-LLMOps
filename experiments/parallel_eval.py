"""Thread-pool helpers for concurrent independent evaluation jobs.

Motivation (2026-07-22): the vLLM client issues one blocking HTTP request
at a time, so a sequential runner drives the server at batch size ~1.
The campaign's jobs -- (dataset x split_seed x decode_seed) evaluations --
are independent by construction: each job builds a FRESH method instance
seeded from its own decode_seed, and each job's internal episode loop
stays sequential, preserving the regression -> reliability-EMA coupling
inside a single evaluate() call.  Pooling jobs across threads therefore
changes nothing but wall time: outputs depend on explicit seeds, never on
scheduling order.

Two hazards handled here:

  * ``config.RANDOM_SEED`` / ``np.random.seed`` are process-global.  They
    are read at construction time (baselines keep their own RandomState
    from then on), so reseed+build -- including lazy component builds via
    ``_ensure_built()`` -- runs under a module lock.
  * vLLM per-request seeds come from each client's instance-local
    ``seed + request_index`` stream, which stays deterministic within its
    own job regardless of interleaving with other jobs.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

_BUILD_LOCK = threading.Lock()


def get_workers(env_var="ASOC_EVAL_WORKERS", default=16):
    """Worker count from env (shared by campaign runners), with floor 1."""
    try:
        return max(1, int(os.environ.get(env_var, str(default))))
    except ValueError:
        return default


def reseed_and_build(seed, build_fn):
    """Set process-global seeds and construct a fresh method instance.

    Serialized under a module lock so concurrent workers cannot interleave
    another job's seed with this job's seed-dependent construction.  If the
    built object exposes ``_ensure_built()`` (pipelines do), it is invoked
    inside the lock as well: component construction (alarm training, KB
    build, encoder load) also reads the global seed.
    """
    import numpy as np
    import config as config_module

    with _BUILD_LOCK:
        config_module.RANDOM_SEED = seed
        np.random.seed(seed)
        obj = build_fn()
        ensure = getattr(obj, "_ensure_built", None)
        if callable(ensure):
            ensure()
        return obj


def iter_job_pool(jobs, workers):
    """Yield ``(key, result)`` in completion order for a list of jobs.

    ``jobs`` is a list of ``(key, fn)`` pairs with ``fn()`` -> result.
    A failing job raises RuntimeError with its key attached, preserving
    the runner's fail-fast behaviour while identifying the job.
    """
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn): key for key, fn in jobs}
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                yield key, fut.result()
            except Exception as exc:
                raise RuntimeError(f"job {key} failed: {exc}") from exc

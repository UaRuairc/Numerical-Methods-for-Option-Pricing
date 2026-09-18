import gc
import random
from collections.abc import Iterable
from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


from src.options import AmericanOption, EuropeanOption
from src.utils import timed
import os
from concurrent.futures import ProcessPoolExecutor




def _axis(v):
    if isinstance(v, (str, bytes)) or not isinstance(v, Iterable):
        return (v,)
    return tuple(v)


def sweep(**axes):
    """Cartesian product over the axes. Scalars count as length-1 axes."""
    axes = {k: _axis(v) for k, v in axes.items()}
    return [dict(zip(axes, vals)) for vals in product(*axes.values())]

def _init_worker():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    gc.disable()


def _one(job):
    cfg, fn, tags = job
    kwargs = {k: v for k, v in cfg.items() if k != "rep"}
    started = datetime.now()
    with timed() as t_ms:
        result = fn(**kwargs)
    return {**cfg, **tags, "method": fn.__name__, "result": result,
            "time_ms": t_ms[0], "started": started}

def run_v2(configs, fn, n_reps=1, shuffle=True, tags=None, save_dir=None,
        filename=None, prefix="results", n_jobs=1, chunksize=None):
    """ parallelised version of run()

        tags are columns that will be added to the dataframe. e.g., {'style':'european}
    """


    tags = tags or {}
    jobs = [{**c, "rep": rep} for rep in range(n_reps) for c in configs]
    if shuffle:
        random.shuffle(jobs)

    if n_jobs == 1:
        rows = []
        gc.disable()
        try:
            for cfg in tqdm(jobs, desc="Running", unit="run"):
                rows.append(_one((cfg, fn, tags)))
        finally:
            gc.enable()
    else:
        workers = os.cpu_count() if n_jobs in (-1, None) else n_jobs

        if chunksize is None:
            chunksize = min(32, max(1, len(jobs) // (workers * 8)))

        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=_init_worker) as ex:
            rows = list(tqdm(ex.map(_one, ((c, fn, tags) for c in jobs),
                                    chunksize=chunksize),
                             total=len(jobs), desc="Running", unit="run"))

    df = pd.DataFrame(rows)
    if save_dir:
        save(df, save_dir, filename, prefix)
    return df

def run(configs, fn, n_reps=1, shuffle=True, tags=None, save_dir=None,
        filename=None, prefix="results"):
    """Call fn(**config) for every config x rep, timing each one.

    tags: constant columns added to every row, not passed to fn.
    """
    tags = tags or {}
    jobs = [{**c, "rep": rep} for rep in range(n_reps) for c in configs]
    if shuffle:
        random.shuffle(jobs)

    rows = []
    gc.disable()
    try:
        for cfg in tqdm(jobs, desc="Running", unit="run"):
            kwargs = {k: v for k, v in cfg.items() if k != "rep"}
            started = datetime.now()
            with timed() as t_ms:
                result = fn(**kwargs)
            rows.append({**cfg, **tags, "method": fn.__name__, "result": result,
                         "time_ms": t_ms[0], "started": started})
    finally:
        gc.enable()

    df = pd.DataFrame(rows)
    if save_dir:
        save(df, save_dir, filename, prefix)
    return df


def save(df, save_dir, filename=None, prefix="results"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    name = filename or f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}"
    path = save_dir / f"{name}.parquet"
    df.to_parquet(path)
    return path


# --- adapters ---------------------------------------------------------------
# One per experiment. For now, will just write a new one rather than adding a flag to an old one.

def cn_european(K, T, contract_type, S0, r, sigma, s_steps, t_steps, version):
    return EuropeanOption(K, T, contract_type).price_CN(
        S0, r, sigma, s_steps, t_steps, version)


def cn_american(K, T, contract_type, S0, r, sigma, s_steps, t_steps, version="v1"):
    return AmericanOption(K, T, contract_type).price_CN(
        S0, r, sigma, s_steps, t_steps, version)


def mc_european(K, T, contract_type, S0, r, sigma, generator, n, seed,
                version="v1", steps=None):
    opt = EuropeanOption(K, T, contract_type)
    extra = {} if steps is None else {"steps": steps}
    return opt.price_MC(S0, r, sigma, generator, version, n=n, seed=seed, **extra)
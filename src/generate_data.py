import pandas as pd
import gc
from src.options import EuropeanOption
import random as pyrandom
from functools import partial
from src.utils import timed
from tqdm.auto import tqdm
from pathlib import Path
from datetime import datetime

def _save_parquet(df, save_dir, filename, prefix):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}"
    path = save_dir / f"{filename}.parquet"
    df.to_parquet(path)
    return path

def _execute_runs(runs, K, T):
    dataframe_rows = []
    total = len(runs)
    gc.disable()
    try:
        with tqdm(total=total, desc="Running", unit="run") as pbar:
            for method, fn in runs:
                started=datetime.now()
                with timed() as t_ms:
                    p = fn()
                dataframe_rows.append({
                    **fn.keywords,
                    'method': method,
                    'K': K,
                    'T': T,
                    'result': p,
                    'time': t_ms[0],
                    'started': started
                })
                pbar.update(1)

    finally:
        gc.enable()
    return dataframe_rows

def generate_mc_data(
        K: float,
        S0: float,
        r: float,
        T: float,
        sigma: float,
        n_reps: int=1,
        steps: tuple=(1000, 2000, 3000, 4000),
        n_paths: tuple=(1000, 2000, 3000, 4000),
        mc_seeds:tuple=tuple(range(10)),
        save: bool=True,
        filename: str=None,
        save_dir: str="./data",
        methods: tuple=('MC_paths_euler', 'MC_paths_milstein', 'MC_exact_integration')
):
    """price european call options using MC methods from ./options.

        Args:
            K: Strike
            S0: Initial spot value
            r: risk free rate
            T: time until maturity
            sigma: vol of gbm process
            n_reps: number of repeated runs (same seed) for analysing hardware effects on time taken

            steps: number of discretised steps for MC methods, assuming MC method requires steps
            n_paths: number of paths to simulate
            mc_seeds: seeds to run for MC methods
            save: save to ./data/mc_results_{timestamp}
            filename: overwrite the default filename scheme
            save_dir: overwrite the default save directory
            methods: which mc methods to generate runs for

        Returns:
            pandas dataframe.
    """
    save_dir = Path(save_dir)
    if save: save_dir.mkdir(parents=True, exist_ok=True)

    call = EuropeanOption(K, T, type="call")
    known_fns = {
        'MC_paths_euler': (call.price_MC_paths_euler, True),
        'MC_paths_milstein': (call.price_MC_paths_milstein, True),
        'MC_exact_integration': (call.price_MC_exact_integration, False),
    }
    known_methods = known_fns.keys()
    unknown = set(methods) - known_methods
    if unknown:
        raise ValueError(f"Unknown mc method(s): {sorted(unknown)}")



    # generate all the runs first then shuffle
    runs=[]

    for m in methods:
        fn, has_steps = known_fns[m]
        for seed in mc_seeds:
            for rep in range(n_reps):
                for n in n_paths:
                    if has_steps:
                        for s in steps:
                            runs.append((m, partial(fn, S0=S0, r=r, sigma=sigma, n_paths=n, steps=s, seed=seed)))
                    else:
                        runs.append((m, partial(fn, S0=S0, r=r, sigma=sigma, n_paths=n, seed=seed)))

    pyrandom.shuffle(runs)

    # execute runs
    dataframe_rows = _execute_runs(runs, K, T)

    # save
    raw = pd.DataFrame(dataframe_rows)
    if save: _save_parquet(raw, save_dir, filename, 'mc_results')

    return raw


def generate_cn_data(
        K: float,
        S0: float,
        r: float,
        T: float,
        sigma: float,
        n_reps: int=1,
        s_steps: tuple=tuple(range(1000,5000,1000)),
        t_steps: tuple=tuple(range(1000,5000,1000)),
        save=True,
        filename: str=None,
        save_dir: str="./data",
        versions: tuple = ('v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7')
):
    """
    generate data using FDM and MC methods from ./options.

        Args:
            K: Strike
            S0: Initial spot value
            r: risk free rate
            T: time until maturity
            sigma: vol of gbm process
            n_reps: number of repeated runs (same seed) for analysing hardware effects on time taken
            s_steps: tuple containing number of steps for S grid for cn methods
            t_steps: number of steps for t grid for FDM (cn only atm) methods
            save: save df
            filename: overwrite the default filename scheme
            save_dir: overwrite the default save directory
            versions: which cn versions to use

        Returns:
            pandas dataframe

        """

    save_dir = Path(save_dir)
    if save: save_dir.mkdir(parents=True, exist_ok=True)

    known_versions = {'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7'}
    unknown = set(versions) - known_versions
    if unknown:
        raise ValueError(f"Unknown cn methods/versions: {sorted(unknown)}")

    print(f"versions chosen: {versions}")
    call = EuropeanOption(K, T, type="call")
    runs = []

    # generate all the runs first then shuffle
    for rep in range(n_reps):
        for s in s_steps:
            for t in t_steps:
                cn = [(f'CN_{v}', partial(call.price_CN, S0=S0, r=r, sigma=sigma, s_steps=s, t_steps=t, version=v)) for v in versions]
                runs.extend(cn)

    pyrandom.shuffle(runs)

    # execute runs
    dataframe_rows = _execute_runs(runs, K, T)

    #save
    raw = pd.DataFrame(dataframe_rows)
    if save: _save_parquet(raw, save_dir, filename, 'cn_results')

    return raw




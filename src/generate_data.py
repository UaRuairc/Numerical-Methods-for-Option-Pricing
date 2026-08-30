import pandas as pd
import gc
from src.options import EuropeanOption
import random as pyrandom
from functools import partial
from src.utils import timed
from tqdm.notebook import tqdm
from pathlib import Path
from datetime import datetime

def _execute_runs(runs, K, T):
    dataframe_rows = []
    total = len(runs)
    gc.disable()
    try:
        with tqdm(total=total, desc="Running", unit="run") as pbar:
            for method, fn in runs:
                p=None
                started=datetime.now()
                with timed() as t_ms:
                    p = fn()
                dataframe_rows.append({
                    'method': method,
                    'K': K,
                    'T': T,
                    **fn.keywords,
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
        filename: str="default",
        save_dir: str="default",
):
    """price european call options using MC methods from ./options.

        Args:
            K: Strike
            S0: Initial spot value
            r: risk free rate
            T: time until maturity
            sigma: variance of gbm process
            n_reps: number of repeated runs (same seed) for analysing hardware effects on time taken

            steps: number of discretised steps for MC methods, assuming MC method requires steps
            n_paths: number of paths to simulate
            mc_seeds: seeds to run for MC methods
            save: save to ./data/mc_results_{timestamp}
            filename: overwrite the default filename scheme
            save_dir: overwrite the default save directory

        Returns:
            pandas dataframe.
    """
    V = EuropeanOption(K, T, type="call")



    # generate all the runs first
    runs=[]
    # mc_methods = ['MC_paths_euler', 'MC_paths_milstein', 'MC_exact_integration']
    for seed in mc_seeds:
        for rep in range(n_reps):
            for n in n_paths:
                runs.append(('MC_exact_integration',
                             partial(V.price_MC_exact_integration, S0=S0, r=r, sigma=sigma,
                                     n_paths=n, seed=seed)))
                for s in steps:
                    runs.append(('MC_paths_euler',
                                 partial(V.price_MC_paths_euler, S0=S0, r=r, sigma=sigma,
                                         n_paths=n, steps=s, seed=seed)))
                    runs.append(('MC_paths_milstein',
                                 partial(V.price_MC_paths_milstein, S0=S0, r=r, sigma=sigma,
                                         n_paths=n, steps=s, seed=seed)))


    pyrandom.shuffle(runs)

    # execute runs
    dataframe_rows = _execute_runs(runs, K, T)

    # save
    raw = pd.DataFrame(dataframe_rows)
    file_ext = '.parquet'

    if save:
        if filename == "default":
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"mc_results_{stamp}"

        if save_dir == "default":
            Path('data').mkdir(exist_ok=True)
            save_dir =  './data/'

        path = f'{save_dir}/{filename}{file_ext}'
        raw.to_parquet(path)

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
        filename: str = "default",
        save_dir: str = "default",
        versions: tuple = ('v1', 'v2', 'v3', 'v4')
):
    """
    generate data using FDM and MC methods from ./options.

        Args:
            K: Strike
            S0: Initial spot value
            r: risk free rate
            T: time until maturity
            sigma: variance of gbm process
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
    print(f"versions chosen: {versions}")
    V = EuropeanOption(K, T, type="call")
    runs = []

    # generate all the runs first and shuffled each block of runs
    for rep in range(n_reps):
        for s in s_steps:
            for t in t_steps:
                cn = [(f'CN_{v}', partial(V.price_CN, S0=S0, r=r, sigma=sigma, s_steps=s, t_steps=t, version=v)) for v in versions]
                runs.extend(cn)

    pyrandom.shuffle(runs)

    # execute runs
    dataframe_rows = _execute_runs(runs, K, T)

    #save
    raw = pd.DataFrame(dataframe_rows)

    file_ext = '.parquet'
    if save:
        if filename == "default":
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"cn_results_{stamp}"

        if save_dir == "default":
            Path('data').mkdir(exist_ok=True)
            save_dir = './data/'

        path = f'{save_dir}/{filename}{file_ext}'
        raw.to_parquet(path)

    return raw




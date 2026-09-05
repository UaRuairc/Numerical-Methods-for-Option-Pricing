import numpy as np


def gbm_exact_integration(
        S0: float,
        r: float,
        T: float,
        sigma: float,
        n: int,
        seed: int = 1,
):
    """
    solves for S by integrating the sde
    """
    rng = np.random.default_rng(seed)
    W = np.sqrt(T) * rng.normal(0, 1, size=n)
    S = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * W)
    return S


def gbm_exact_integration_reconstruct(
        S0: float,
        r: float,
        T: float,
        sigma: float,
        n: int,
        steps: int,
        seed: int = 1,
):
    """
    In order to test strong order error, must reconstruct W from the same dW used to estimate pathwise methods
    """
    rng = np.random.default_rng(seed)
    dt = T / steps
    dW = np.sqrt(dt) * rng.normal(0, 1, size=(n, steps))
    W = dW.sum(axis=1)
    S = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * W)

    return S


def gbm_paths_euler(
        S0: float,
        r: float,
        T: float,
        sigma: float,
        n: int,
        steps: int,
        seed: int = 1
):
    """
    solves for S by discretising the sde
    """
    rng = np.random.default_rng(seed)
    dt = T / steps
    dW = np.sqrt(dt) * rng.normal(0, 1, size=(n, steps))
    S = np.zeros((n, steps + 1))
    S[:, 0] = S0
    for t in range(steps):
        S[:, t + 1] = S[:, t] * (1 + r * dt + sigma * dW[:, t])
    return S


def gbm_paths_milstein(
        S0: float,
        r: float,
        T: float,
        sigma: float,
        n: int,
        steps: int,
        seed: int = 1
):
    """
    solves for S by discretising the sde
    """
    rng = np.random.default_rng(seed)
    dt = T / steps
    dW = np.sqrt(dt) * rng.normal(0, 1, size=(n, steps))
    S = np.zeros((n, steps + 1))
    S[:, 0] = S0
    for t in range(steps):
        S[:, t + 1] = S[:, t] * (1 + r * dt + sigma * dW[:, t] + 0.5 * sigma ** 2 * (dW[:, t] ** 2 - dt))
    return S


def strong_error(fn, S0, r, T, sigma, n, steps, seed):
    S_T = fn(S0, r, T, sigma, n=n, steps=steps, seed=seed)[:, -1]
    S_exact = gbm_exact_integration_reconstruct(S0, r, T, sigma, n=n, steps=steps, seed=seed)
    return np.abs(S_T - S_exact).mean()

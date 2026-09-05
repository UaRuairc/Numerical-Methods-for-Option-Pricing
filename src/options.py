import numpy as np
import scipy.stats
from typing import Literal

from src.fdm import pde_crank_nicolson, pde_crank_nicolson_v2, pde_crank_nicolson_v3, pde_crank_nicolson_v4, \
    pde_crank_nicolson_v5, pde_crank_nicolson_v6, pde_crank_nicolson_v7
from src.monte_carlo import gbm_exact_integration, gbm_exact_integration_reconstruct, gbm_paths_euler, \
    gbm_paths_milstein


def BS_closed_form(S0, K, r, T, sigma):
    """
    Uses the exact solution to Black Scholes PDE to return the discounted expected intrinsic value of a European call option
    """
    d1 = 1 / (sigma * np.sqrt(T)) * (np.log(S0 / K) + (r + sigma ** 2 / 2) * T)
    d2 = d1 - sigma * np.sqrt(T)
    N1 = scipy.stats.norm.cdf(d1)
    N2 = scipy.stats.norm.cdf(d2)
    V = N1 * S0 - N2 * K * np.exp(-r * T)
    price = float(V)
    return price


class EuropeanOption:
    def __init__(self, K: float, T: float, type: Literal["call", "put"]):
        self.K = K
        self.T = T
        self.style = "European"
        self.type = type

    def payoff(self, S):
        return np.maximum(S - self.K, 0)

    def discount(self, x, r):
        return x * np.exp(- r * self.T)

    def put_call_parity(self, call, S0, r):
        return call - S0 + self.discount(self.K, r)

    def price(self, call, S0, r):
        return call if self.type == "call" else self.put_call_parity(call, S0, r)

    def price_closed_form(self, S0, r, sigma):
        return self.price(BS_closed_form(S0, self.K, r, self.T, sigma), S0, r)

    def price_MC_exact_integration(self, S0, r, sigma, n_paths=10000, seed=1):
        S = gbm_exact_integration(S0, r, self.T, sigma=sigma, n=n_paths, seed=seed)
        payoffs = self.payoff(S)
        call = self.discount(payoffs.mean(), r)
        return self.price(call, S0, r)

    def price_MC_exact_integration_reconstruct(self, S0, r, sigma, n_paths=10000, steps=100, seed=1):
        """ to test strong order """
        S = gbm_exact_integration_reconstruct(S0, r, self.T, sigma=sigma, n=n_paths, steps=steps, seed=seed)
        payoffs = self.payoff(S)
        call = self.discount(payoffs.mean(), r)
        return self.price(call, S0, r)

    def price_MC_paths_euler(self, S0, r, sigma, n_paths=10000, steps=100, seed=1):
        paths = gbm_paths_euler(S0, r, self.T, sigma, n=n_paths, steps=steps, seed=seed)
        S = paths[:, -1]
        payoffs = self.payoff(S)
        call = self.discount(payoffs.mean(), r)
        return self.price(call, S0, r)

    def price_MC_paths_milstein(self, S0, r, sigma, n_paths=10000, steps=100, seed=1):
        paths = gbm_paths_milstein(S0, r, self.T, sigma, n=n_paths, steps=steps, seed=seed)
        S = paths[:, -1]
        payoffs = self.payoff(S)
        call = self.discount(payoffs.mean(), r)
        return self.price(call, S0, r)

    def price_CN(self, S0, r, sigma, s_steps=100, t_steps=500, version : str="v1", return_greeks=False):

        solvers = {
            "v1": pde_crank_nicolson,        # initial version made: unoptimised & stores V for all t_steps. Uses scipy's splu(L) for LU factorisation
            "v2": pde_crank_nicolson_v2,     # modified v1 to only store the most recent t_step. Still uses splu(L).
            "v3": pde_crank_nicolson_v3,     # modified v2 to use scipy's LAPACK for tridiagonal solve instead of scipy's splu. Also no longer define R.
            "v4": pde_crank_nicolson_v4,     # modified v3 to update V in-place, and also introduce Rannacher smoothing. Still uses LAPACK.
            "v5": pde_crank_nicolson_v5,     # modified v4 to use custom tridiagonal solve that can be JIT compiled using numba.
            "v6": pde_crank_nicolson_v6,     # modified v5 to JIT compile the whole cn loop, not just tridiagonal solve.
            "v7": pde_crank_nicolson_v7,     # modified v6 to compute d_inverse outside the loop and use  multiplication by d_inverse during back substitution instead
        }

        solver = solvers[version]

        V, grid = solver(self.K, S0, r, self.T, sigma, s_steps=s_steps, t_steps=t_steps)

        x = np.log(grid['S'])

        call = np.interp(np.log(S0), x, V)

        if return_greeks:
            delta, gamma, theta = self._greeks_CN(V, x, sigma, r)
            return self.price(call, S0, r), delta, gamma, theta, x

        return self.price(call, S0, r)

    def _greeks_CN(self, V, x, sigma, r):
        """
        Compute the greeks by differencing the final time slice V, where V is computed by a CN method.
        """
        # d/dS = (1 / S) * d/dx
        # delta = dV/dS = (1 / S) dV/dx
        # gamma = d^2V/dS^2 = d/dS ( (1 / S) * dV/dx ) = (1/S^2) ( d^2V/dx^2 - dV/dx )

        dx = x[1] - x[0]
        Vx = (V[2:] - V[:-2]) / (2 * dx)
        Vxx = (V[2:] - 2 * V[1:-1] + V[:-2]) / (dx) ** 2

        S = np.exp(x[1:-1])  # S = exp(x)
        delta = Vx / S
        gamma = (Vxx - Vx) / (S ** 2)

        # dv/dtau = 1/2 sigma^2 d^2V/dx^2 + (r-1/2 sigma^2)dV/dx - rV
        # theta = - dv/dtau
        theta = -0.5 * sigma ** 2 * Vxx - (r - 0.5 * sigma ** 2) * Vx + r * V[1:-1]

        if self.type == 'put':
            # use put call parity:

            # P = C - S + e^(-r*(T-t)) * K

            # d/dS P = d/dS C - 1 + 0
            delta = delta - 1

            # d/dS Delta_P = d/dS Delta_C - 0
            # gamma = gamma

            # d/dt P = d/dt C - 0 + r * e^(-r*(T-t)) * K
            theta = theta + r * self.discount(self.K, r)

        return delta, gamma, theta


if __name__ == "__main__":
    print("hi")
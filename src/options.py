import numpy as np
import scipy.stats
from typing import Literal

from src.fdm import get_fdm_solver
from src.monte_carlo import get_mc_generator

def get_solver(scheme):
    return {'MC': get_mc_generator, 'CN': get_fdm_solver}[scheme]

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
    def __init__(self, K: float, T: float, contract_type: Literal["call", "put"]):
        self.K = K
        self.T = T
        self.contract_type = contract_type

    def payoff(self, S):
        if self.contract_type == "call":
            return np.maximum(S - self.K, 0)
        return np.maximum(self.K - S, 0)

    def discount(self, x, r):
        return x * np.exp(- r * self.T)

    def put_call_parity(self, call, S0, r):
        return call - S0 + self.discount(self.K, r)

    def price_closed_form(self, S0, r, sigma):
        call_price = BS_closed_form(S0, self.K, r, self.T, sigma)
        if self.contract_type != "call":
            return self.put_call_parity(call_price, S0, r)
        return call_price

    def price(self, scheme, **kwargs):

        if scheme == "MC":
            return self.price_MC(**kwargs)
        elif scheme == "CN":
            return self.price_CN(**kwargs)
        else:
            raise NotImplementedError(f"{scheme} not implemented")

    def price_MC(self, S0, r, sigma, generator, version="v1", **kwargs):
        gen = get_mc_generator(generator, version)
        result = gen(S0=S0, r=r, T=self.T, sigma=sigma, **kwargs)
        S = result if result.ndim == 1 else result[:, -1]
        return self.discount(self.payoff(S).mean(), r)

    def price_CN(self, S0, r, sigma, s_steps=100, t_steps=500, version : str="v7", return_greeks=False):

        """
            FDM_SOLVERS["pde_crank_nicolson"]["european"] = {
                "v1": pde_crank_nicolson,        # initial version made: unoptimised & stores V for all t_steps. Uses scipy's splu(L) for LU factorisation
                "v2": pde_crank_nicolson_v2,     # modified v1 to only store the most recent t_step. Still uses splu(L).
                "v3": pde_crank_nicolson_v3,     # modified v2 to use scipy's LAPACK for tridiagonal solve instead of scipy's splu. Also no longer define R.
                "v4": pde_crank_nicolson_v4,     # modified v3 to update V in-place, and also introduce Rannacher smoothing. Still uses LAPACK.
                "v5": pde_crank_nicolson_v5,     # modified v4 to use custom tridiagonal solve that can be JIT compiled using numba.
                "v6": pde_crank_nicolson_v6,     # modified v5 to JIT compile the whole cn loop, not just tridiagonal solve.
                "v7": pde_crank_nicolson_v7,     # modified v6 to compute d_inverse outside the loop and use  multiplication by d_inverse during back substitution instead
            }
        """

        solver = get_fdm_solver(method="pde_crank_nicolson", style="european", version=version)

        V, grid = solver(self.K, S0, r, self.T, sigma, s_steps=s_steps, t_steps=t_steps)

        x = np.log(grid['S'])

        call_price = np.interp(np.log(S0), x, V)

        if self.contract_type != "call":
            price = self.put_call_parity(call_price, S0, r)
        else:
            price = call_price


        if return_greeks:
            delta, gamma, theta = self._greeks_CN(V, x, sigma, r)
            return price, delta, gamma, theta, x

        return price

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

        if self.contract_type == 'put':
            # use put call parity:

            # P = C - S + e^(-r*(T-t)) * K

            # d/dS P = d/dS C - 1 + 0
            delta = delta - 1

            # d/dS Delta_P = d/dS Delta_C - 0
            # gamma = gamma

            # d/dt P = d/dt C - 0 + r * e^(-r*(T-t)) * K
            theta = theta + r * self.discount(self.K, r)

        return delta, gamma, theta




class AmericanOption:
    def __init__(self, K, T, contract_type="call"):
        self.K = K
        self.T = T
        self.contract_type = contract_type

    def price_CN(self, S0, r, sigma, s_steps, t_steps, version = "v1"):
        """
            FDM_SOLVERS["american"] = {
                "v1": pde_american_crank_nicolson,        # initial version made: similar level of optimsation as pde_crank_nicolson_v7
            }
        """
        solver = get_fdm_solver(method="pde_crank_nicolson", style="american", version=version)

        V, grid = solver(self.K, S0, r, self.T, sigma, s_steps, t_steps, contract_type=self.contract_type)
        x = np.log(grid['S'])
        option_price = np.interp(np.log(S0), x, V)
        return option_price

"""
    def _price_CN(self, S0, r, sigma, s_steps, t_steps):
        x, dx, tau, dt, n_low = build_grid(self.K, S0, r, self.T, sigma, s_steps, t_steps, n_std=5)
        grid = (x, dx, tau, dt)
        V = pde_crank_nicolson_american(self.K, r, sigma, grid)
        option_price = np.interp(np.log(S0), x, V)
"""
if __name__ == "__main__":
    print("hi")
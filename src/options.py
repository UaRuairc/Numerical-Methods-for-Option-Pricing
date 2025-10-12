import numpy as np
from numpy import random
import scipy.stats
from time import perf_counter
from contextlib import contextmanager


class EuropeanOption:

    def __init__(self, sigma, r, S0, T, K, iter=10000, timesteps=100):

        self.sigma = sigma
        self.r = r
        self.S0 = S0
        self.T = T
        self.timesteps = timesteps
        self.K = K
        self.iter = iter
        self.discount_factor = np.exp(-self.r*self.T)
        self.prices = {}

    def print_params(self):

        params = {
            'sigma': self.sigma,
            'r': self.r,
            'S_0': self.S0,
            'T': self.T,
            'K': self.K,
            'iter': self.iter,
            'timesteps': self.timesteps
        }
        print("\nParameters:")
        for key, val in params.items():
            print(f"  {key}: {val}")

    def price(self, methods):


        for method in methods:
            self.prices.setdefault(method, {"call": None, "put": None})
            fn = getattr(self, method, None)
            if not callable(fn):
                print(f"{method} is not a valid method.")
                continue

            with timed(method):
                discounted_expected_intrinsic_value = fn()
            self.prices[method]["call"] = discounted_expected_intrinsic_value
            self.prices[method]["put"] = self.put_call_parity(discounted_expected_intrinsic_value)

        return [self.prices.get(method, None) for method in methods]

    def put_call_parity(self, call):
        put = call - self.S0 + self.discount_factor * self.K
        return put.item()

    def sde_exact_mc(self):

        """
        Uses the exact closed form solution of the SDE to return the expected discounted expected intrinsic value of a European call option
        """

        W =  np.sqrt(self.T) * random.normal(0, 1, size=self.iter)

        S = self.S0 * np.exp((self.r - 0.5*self.sigma**2)*self.T + self.sigma*W)

        spot_minus_k = S - self.K

        discounted_expected_intrinsic_value = self.discount_factor*(spot_minus_k[spot_minus_k>0].sum())/self.iter

        return discounted_expected_intrinsic_value.item()

    def sde_euler_maruyama_mc(self):

        """
        Uses  method to solve the SDE equation and returns the expected discounted expected intrinsic value of a European call option
        """

        dt = self.T / self.timesteps
        dW = np.sqrt(dt) * random.normal(0, 1, size=(self.iter, self.timesteps))
        S = np.zeros((self.iter, self.timesteps + 1))
        S[:,0] = self.S0

        for t in range(self.timesteps):
            S[:, t+1] = S[:, t] * (1 + self.r * dt + self.sigma * dW[:, t])

        S_t = S[:, -1]

        spot_minus_k = S_t - self.K

        discounted_expected_intrinsic_value = self.discount_factor * ( spot_minus_k[spot_minus_k > 0].sum() )/self.iter

        return discounted_expected_intrinsic_value.item()

    def pde_exact(self, type="call"):
        """
        Uses the exact solution to Black Scholes PDE to return the discounted expected intrinsic value of a European call option
        """
        d1 = 1 /(self.sigma*np.sqrt(self.T)) * (np.log(self.S0/self.K) + (self.r + self.sigma**2/2)*self.T)
        d2 = d1 - self.sigma * np.sqrt(self.T)

        N1 = scipy.stats.norm.cdf(d1)
        N2 = scipy.stats.norm.cdf(d2)

        discounted_expected_intrinsic_value = N1 * self.S0 - N2 * self.K * np.exp(-self.r*self.T)

        return discounted_expected_intrinsic_value.item()

    def pde_fdm(self, type="call"):
        pass


@contextmanager
def timed(name="method"):
    start = perf_counter()
    try:
        yield
    finally:
        end = perf_counter()
        print(f"{name} took {end - start:.4f}s")





if __name__ == "__main__":

    V = EuropeanOption(sigma=0.06, r=0.04, S0=100, T=1, K=110, iter=10000, timesteps=100)

    methods = ["pde_exact", "sde_exact_mc", "sde_euler_maruyama_mc"]


    V.print_params()
    V.price(methods)

    print("\nBS prices: ")

    for method in methods:
        print(f"\nMethod: {method}")
        print(f"Prices: {V.prices[method]}")
        if method != "pde_exact":
            print(f"CALL ERROR: {(V.prices["pde_exact"]["call"] - V.prices[method]["call"]) / V.prices["pde_exact"]["call"] * 100:.4f}%")
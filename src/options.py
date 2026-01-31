import numpy as np
from numpy import random
import scipy.stats
from time import perf_counter
from contextlib import contextmanager
import matplotlib.pyplot as plt

from scipy.sparse import diags
from scipy.sparse.linalg import splu


class EuropeanOption:

    def __init__(self, sigma, r, S0, T, K, mc_paths=10000, t_steps=100, s_steps=500):

        self.sigma = sigma
        self.r = r
        self.S0 = S0
        self.T = T
        self.K = K

        self.mc_paths = mc_paths
        self.t_steps = t_steps
        self.s_steps = s_steps


        self.discount_factor = np.exp(-self.r*self.T)
        self.prices = {}

    def print_params(self):

        params = {
            'sigma': self.sigma,
            'r': self.r,
            'S_0': self.S0,
            'T': self.T,
            'K': self.K,
            'mc_paths': self.mc_paths,
            't_steps': self.t_steps,
            's_steps': self.s_steps,
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
        return float(put)

    def sde_exact_mc(self):

        """
        Uses the exact closed form solution of the SDE to return the expected discounted expected intrinsic value of a European call option
        """

        W =  np.sqrt(self.T) * random.normal(0, 1, size=self.mc_paths)

        S = self.S0 * np.exp((self.r - 0.5*self.sigma**2)*self.T + self.sigma*W)

        spot_minus_k = S - self.K

        discounted_expected_intrinsic_value = self.discount_factor*(spot_minus_k[spot_minus_k>0].sum())/self.mc_paths

        return discounted_expected_intrinsic_value.item()

    def sde_euler_maruyama_mc(self):

        """
        Uses  method to solve the SDE equation and returns the expected discounted expected intrinsic value of a European call option
        """

        dt = self.T / self.t_steps
        dW = np.sqrt(dt) * random.normal(0, 1, size=(self.mc_paths, self.t_steps))
        S = np.zeros((self.mc_paths, self.t_steps + 1))
        S[:,0] = self.S0

        for t in range(self.t_steps):
            S[:, t+1] = S[:, t] * (1 + self.r * dt + self.sigma * dW[:, t])

        S_t = S[:, -1]

        spot_minus_k = S_t - self.K

        discounted_expected_intrinsic_value = self.discount_factor * ( spot_minus_k[spot_minus_k > 0].sum() )/self.mc_paths

        return discounted_expected_intrinsic_value.item()

    def pde_exact(self):
        """
        Uses the exact solution to Black Scholes PDE to return the discounted expected intrinsic value of a European call option
        """
        d1 = 1 /(self.sigma*np.sqrt(self.T)) * (np.log(self.S0/self.K) + (self.r + self.sigma**2/2)*self.T)
        d2 = d1 - self.sigma * np.sqrt(self.T)

        N1 = scipy.stats.norm.cdf(d1)
        N2 = scipy.stats.norm.cdf(d2)

        discounted_expected_intrinsic_value = N1 * self.S0 - N2 * self.K * np.exp(-self.r*self.T)

        return discounted_expected_intrinsic_value.item()

    def pde_crank_nicolson(self, min_strike, max_strike, dK, S0=None, s_steps=None, timesteps=None, T=None):
        """
        the crank nicolson method to solve the Black Scholes PDE for a range of strikes, and a plot of the option surface
        """

        """
        
        In linear space, the PDE is:
                 dv/dt + 1/2 sigma^2 S^2 d^2v/dS^2 + rS dv/dS - rv = 0      
        we set tau = T - t, thus
                 dv/dtau = 1/2 sigma^2 S^2 d^2v/dS^2 + rS dv/dS - rv

        In log space, x = ln(S), the PDE is:                                    <------------ we use this
                 dv/dtau = 1/2 sigma^2 d^2v/dx^2 + (r-1/2 sigma^2)dv/dx - rv

        Boundary conditions:
        At maturity V = S-K                  => for all i, V[i,j=0]  = max(S-K, 0)
        Deep OTM options V = 0               => for all j, V[i=0,j]  = 0
        Deep ITM options V = S - K exp(-rt)  => for all j, V[i=-1,j] = S_max - K exp(-rt)
                 
        """

        S0, s_steps, timesteps, T = (
            self.S0 if S0 is None else S0,
            self.s_steps if s_steps is None else s_steps,
            self.t_steps if timesteps is None else timesteps,
            self.T if T is None else T
        )

        i = s_steps + 1
        j = timesteps + 1

        low = min(S0, min_strike)
        high = max(S0, max_strike)

        #print(low)
        #print(high)

        logS_min = np.log(low) - 5 * self.sigma * np.sqrt(T)
        logS_max = np.log(high) + 5 * self.sigma * np.sqrt(T)

        #print(logS_min)
        #print(logS_max)
        #print(np.exp(logS_min), np.exp(logS_max))
        logS = np.linspace(logS_min, logS_max, i)

        S = np.exp(logS)

        dx = (logS_max - logS_min) / (i-1)
        dt = T / (j-1)

        """
        PDE:
        dv/dtau = 1/2 sigma^2 d^2v/dx^2 + (r-1/2 sigma^2)dv/dx - rv
             
        Crank Nicolson method in log space:
        The terms in their differenced forms (implicit - explicit):
        dv/dt     = v^n+1_i - v^n_i / dt
        d^2v/dS^2 = (1/2)(v^n_i+1 - 2v^n_i + v^n_i-1) / dS^2 + (1/2)(v^n+1_i+1 - 2v^n+1_i + v^n+1_i-1) / dS^2
        dv/dS     = (1/2)(v^n_i+1 - v^n_i-1) / dS + (1/2)(v^n+1_i+1 - v^n+1_i-1) / dS
        
        The rv term is also differenced as r (v^n+1_i + v^n_i) / 2
        
        CN FDM:

        dv/dtau   = v[i, n+1] - v[i, n] / dtau 
        d^2v/dx^2 = [ (1/2)(v[i+1, n] - 2v[i, n]+ v[i-1, n])   + (1/2)(v[i+1, n+1] - 2v[i, n+1]+ v[i-1, n+1]) ] / dx^2   
        dv/dx  =    [ (1/2)(v[i+1, n]- v[i-1,n])               + (1/2)(v[i+1, n+1]- v[i-1, n+1])              ] / 2dx  
        v =         [ (1/2)(v[i, n])                           + (1/2)(v[i, n+1])                             ]

        Define:
        n = 1, 2, ..., self.t_steps + 1        [ index for tau        ]
        i = 1, 2, ..., S_steps + 1             [ index for step count ]

        Thus:
        v[i, n+1] - v[i, n] / dtau =

        + (1/2 sigma^2)   [   (1/2)(v[i+1, n] - 2v[i, n]+ v[i-1, n])   + (1/2)(v[i+1, n+1] - 2v[i, n+1]+ v[i-1, n+1]) ] / dx^2   
        + (r-1/2 sigma^2) [   (1/2)(v[i+1, n]- v[i-1,n])               + (1/2)(v[i+1, n+1]- v[i-1, n+1])              ] / 2dx  
        - r               [   (1/2)(v[i, n])                           + (1/2)(v[i, n+1])                             ]

        multiply by dtau, then collect all t = n+1 terms on the LHS, and t = n terms on the RHS

        define a = dtau * (1/2 sigma^2)   / dx^2  
               b = dtau * (r-1/2 sigma^2) / 2 dx
               c = dtau * (-r)
        """

        a = dt * (0.5 * self.sigma ** 2) / dx ** 2
        b = dt * (self.r - 0.5 * self.sigma ** 2) / (2 * dx)
        c = dt * (-self.r)

        """

        LHS = 2 v[i, n+1]   - a [ v[i+1, n+1] - 2v[i, n+1] + v[i-1, n+1] ] / dx^2   
                            - b [ v[i+1, n+1] - v[i-1, n+1]              ] / dx  
                            - c [ v[i, n+1]                              ]

        RHS = 2 v[i, n]     + a [ v[i+1, n] - 2v[i, n]+ v[i-1, n]        ] / dx^2
                            + b [ v[i+1, n] - v[i-1, n]                  ] / dx
                            + c [ v[i, n]                                ]

        Rearranging:

        LHS = - (a+b) * v[i+1, n+1]    + (2 + (2a - c)) * v[i, n+1]    - (a-b) * v[i-1, n+1]
        RHS = + (a+b) * v[i+1, n]      + (2 - (2a - c)) * v[i, n]      + (a-b) * v[i-1, n]

        redefine A = - (a-b)
                 B = 2a - c
                 C = - (a+b)
                 
        """

        A = -(a - b)
        B = 2 * a - c
        C = -(a + b)

        """

        LHS =   A * v[i-1, n+1]    + (2 + B) * v[i, n+1]    + C * v[i+1, n+1] 
        RHS = - A * v[i-1, n]      + (2 - B) * v[i, n]      - C * v[i+1, n]      

        My derivation has a 2 +/- B rather than 1 +/ B. This is still correct, I just essentially multiplied through by 2
        earlier. Thus, my A, B and C are twice the size of typical derivations.

        Since we remove the boundaries, we will actually have
        
        L v[1:-1, n+1] = R v[1:-1, n] + boundary conditions
        
        L = tri_diag[ A        (2+B)        C]
        R = tri_diag[-A        (2-B)       -C]
        
        we will need L and R to be of dimension: i - 2
           
        The off diagonal terms lose a dimension due to the tridiagonal nature of the matrix, thus we create arrays
        A, B and C such that:
        dim(A) = dim(C) = i - 3
        dim(B) = i - 2
        """

        A_ = A * np.ones(i - 3)
        B_left_ = (2 + B) * np.ones(i - 2)
        B_right_ = (2 - B) * np.ones(i - 2)
        C_ = C * np.ones(i - 3)

        L = diags([A_, B_left_, C_], [-1, 0, 1]).tocsc()
        R = diags([-A_, B_right_, -C_], [-1, 0, 1]).tocsc()

        """
        LU factorisation can be used to solve the linear system. 
        The Thomas algorithm can also be used (a forward pass to eliminate the lower diagonal, 
        then a backward pass to solve for the unknowns)
        
        We use LU factorisation here. In log space it's constant for all strikes
        """

        LU = splu(L)

        """
        Reminder: for a given strike, K:

        The final time slice of V, at j=-1, is the option price at T = 0. For each strike K, we have
        
        V_S0[K, :] = V[:, -1]

        """

        number_of_strikes = int(np.floor((max_strike - min_strike) / dK)) + 1

        V_S0 = np.zeros((number_of_strikes, self.s_steps + 1))
        strikes = np.linspace(min_strike, max_strike, number_of_strikes)

        """
        Boundary conditions:
        At maturity V = S-K                  => for all i, V[i,j=0]  = max(S-K, 0)
        Deep OTM options V = 0               => for all j, V[i=0,j]  = 0
        Deep ITM options V = S - K exp(-rt)  => for all j, V[i=-1,j] = S_max - K exp(-rt)
        """

        linear_grid = np.exp(np.linspace(logS_min, logS_max, i))

        for k_idx in range(number_of_strikes):
            K = min_strike + k_idx * dK
            V = np.zeros((i, j))
            V[:, 0] = np.maximum(linear_grid - K, 0)
            V[0, :] = np.zeros(j)
            V[-1, :] = np.exp(logS_max) - K * np.exp(-self.r * np.linspace(0, T, j))

            #for n in trange(timesteps, desc="Time step"):
            for n in range(timesteps):
                """
                L v[1:-1, n+1] = R v[1:-1, n] + boundary conditions
                """

                rhs = R @ V[1:-1,n]
                rhs[0] -= A * (V[0, n] + V[0, n+1])
                rhs[-1] -= C * (V[-1, n] + V[-1, n+1])
                V[1:-1, n+1] = LU.solve(rhs)


            V_S0[k_idx, :] =  V[:, -1]


        self.plot_option_surface(strikes, S, V_S0)

        return None





    def plot_option_surface(self, strikes, S, V_S0):

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        X, Y = np.meshgrid(strikes, S, indexing='ij')
        surf = ax.plot_surface(X, Y, V_S0, cmap='viridis')
        ax.set_xlabel('Strike')
        ax.set_ylabel('Underlying S')
        ax.set_zlabel('Price')
        ax.view_init(elev=25, azim=-25)
        #fig.colorbar(surf, shrink=0.6, aspect=12, label='Price')
        plt.show()




@contextmanager
def timed(name="method"):
    start = perf_counter()
    try:
        yield
    finally:
        end = perf_counter()
        print(f"{name} took {end - start:.4f}s")





if __name__ == "__main__":

    V = EuropeanOption(sigma=0.2, r=0.04, S0=100, T=1, K=110, mc_paths=10000, t_steps=100, s_steps=500)

    # methods = ["pde_exact", "sde_exact_mc", "sde_euler_maruyama_mc"]

    """
    V.print_params()
    V.price(methods)

    print("\nBS prices: ")

    for method in methods:
        print(f"\nMethod: {method}")
        print(f"Prices: {V.prices[method]}")
        if method != "pde_exact":
            print(f"CALL ERROR: {(V.prices["pde_exact"]["call"] - V.prices[method]["call"]) / V.prices["pde_exact"]["call"] * 100:.4f}%")
    """

    V.pde_crank_nicolson(min_strike=90, max_strike=110, dK=2)
    #V.option_surface(min_strike=80, max_strike=120, dK=2)
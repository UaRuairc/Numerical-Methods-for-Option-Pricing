import numpy as np
import scipy.stats
from typing import Literal
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from scipy.linalg.lapack import dgttrf, dgttrs
# from numba import njit

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

def build_grid(
        K,
        S0,
        r,
        T,
        sigma,
        s_steps,
        t_steps,
        n_std,
):


    vol = sigma * np.sqrt(T)
    logK, logS0 = np.log(K), np.log(S0)

    # initialise upper and lower bounds to guard against a low probability gbm path
    low = n_std * vol + (r - 0.5 * sigma ** 2) * T
    high = n_std * vol - (r + 0.5 * sigma ** 2) * T

    # make sure S0 and K are inside the grid.
    # guard against large maturity T (or large sigma/r)
    low = max(0.7 * n_std * vol, low)
    high = max(0.7 * n_std * vol, high)

    dx = (low+high)/s_steps

    # make log(S) grid include K

    n_low = int(np.ceil(low/dx))
    n_high = s_steps - n_low
    x = logK + dx * np.arange(-n_low, n_high+1)

    n_t = t_steps + 1
    tau, dt = np.linspace(0, T, n_t, retstep=True)

    return x, dx, tau, dt



def pde_crank_nicolson(
        K,
        S0,
        r,
        T,
        sigma,
        s_steps=100,
        t_steps=100,
        return_full_V=False
):
    """
    the crank nicolson method to solve the Black Scholes PDE for a range of strikes, and a plot of the option surface
    for multiple strikes. This is not an efficient
    """

    """
    
    !!!!!!!!!!
    # NOTE: IN THIS DERIVATION USES V[x,t] BUT IN THE CODE WE WILL USE V[t,x]
    !!!!!!!!!!
    
    In linear space, the PDE is:
             dv/dt + 1/2 sigma^2 S^2 d^2v/dS^2 + rS dv/dS - rv = 0      
    we set tau = T - t, thus
             dv/dtau = 1/2 sigma^2 S^2 d^2v/dS^2 + rS dv/dS - rv

    In log space, x = ln(S), the PDE is:                                    
             dv/dtau = 1/2 sigma^2 d^2v/dx^2 + (r-1/2 sigma^2)dv/dx - rv        (Eq.1)
    
    
    1. We denote V[i, n] = V[x_i, tau_n]. In the code we will also have a third index, k: V[i, n, k] where k is the index for the strike.
    
    2. We denote spatial steps by dx, and temporal steps by dt = dtau
    
    Boundary conditions
    At maturity V = S-K                  => for all grid points at maturity, V[i,n=0]  = max(S-K, 0)
    Deep OTM options V = 0               => for all time steps tau_n, V[i=0,n]  = 0
    Deep ITM options V = S - K exp(-rt)  => for all time steps tau_n, V[i=-1,j] = S_max - K exp(-rt)

    """


    # Grid
    x, dx, tau, dt = build_grid(K, S0, r, T, sigma, s_steps, t_steps, n_std=5)
    S = np.exp(x)

    # ------------------------------------------------------------------
    # recall: typically one writes v(x_i,t_n) = v[i, n] but for speed in numpy it's better to have v(t_n,x_i) since we move forward in t
    # ------------------------------------------------------------------


    n_x = s_steps + 1
    n_t = t_steps + 1

    V = np.zeros((n_t, n_x))

    """
    Define:
    n = 1, 2, ..., t_steps + 1             [ index for tau        ]
    i = 1, 2, ..., S_steps + 1             [ index for step count ]
    Crank Nicolson method in log space:

    The various terms in their differenced forms (implicit - explicit for x coordinate, implicit in time):

    dv/dtau   = v[i, n+1] - v[i, n] / dtau                                                                                
    d^2v/dx^2 = [ (1/2)(v[i+1, n] - 2v[i, n]+ v[i-1, n])   + (1/2)(v[i+1, n+1] - 2v[i, n+1]+ v[i-1, n+1]) ] / dx^2   
    dv/dx     = [ (1/2)(v[i+1, n]- v[i-1,n])               + (1/2)(v[i+1, n+1]- v[i-1, n+1])              ] / 2dx  
    v         = [ (1/2)(v[i, n])                           + (1/2)(v[i, n+1])                             ]

    Substituting above into (Eq.1) we get:

    v[i, n+1] - v[i, n] / dtau =                                                                                          (Eq.2)
    + (1/2 sigma^2)   [ (1/2)(v[i+1, n] - 2v[i, n]+ v[i-1, n])   + (1/2)(v[i+1, n+1] - 2v[i, n+1]+ v[i-1, n+1]) ] / dx^2      
    + (r-1/2 sigma^2) [ (1/2)(v[i+1, n]- v[i-1,n])               + (1/2)(v[i+1, n+1]- v[i-1, n+1])              ] / 2dx  
    - r               [ (1/2)(v[i, n])                           + (1/2)(v[i, n+1])                             ]

    define a = dtau * (1/2 sigma^2)   / dx^2  = sigma^2         * (dtau/2dx^2)
           b = dtau * (r-1/2 sigma^2) / 2 dx  = (r-1/2 sigma^2) * (dtau/2dx)
           c = dtau * (-r)
    """

    a = dt * (0.5 * sigma ** 2) / dx ** 2
    b = dt * (r - 0.5 * sigma ** 2) / (2 * dx)
    c = -r * dt

    """
    multiply (Eq.2) by dtau, then collect all t = n+1 terms on the LHS, and t = n terms on the RHS. And substitute in a, b, c.
    LHS = 2 v[i, n+1]   - a [ v[i+1, n+1] - 2v[i, n+1] + v[i-1, n+1] ]
                        - b [ v[i+1, n+1] - v[i-1, n+1]              ]
                        - c [ v[i, n+1]                              ]
    
    RHS = 2 v[i, n]     + a [ v[i+1, n] - 2v[i, n]+ v[i-1, n]        ]
                        + b [ v[i+1, n] - v[i-1, n]                  ]
                        + c [ v[i, n]                                ]
    Collecting terms at the same spatial coord,
    LHS = - (a+b) * v[i+1, n+1]    + (2 + (2a - c)) * v[i, n+1]    - (a-b) * v[i-1, n+1]
    RHS = + (a+b) * v[i+1, n]      + (2 - (2a - c)) * v[i, n]      + (a-b) * v[i-1, n]
    
    redefine A = - (a-b)
            B = 2a - c
            C = - (a+b)
    (note in this derivation my coefficients absorbed an additional factor of 2)
    """
    A = -(a - b)
    B = 2 * a - c
    C = -(a + b)

    """
        LHS =   A * v[i-1, n+1]    + (2 + B) * v[i, n+1]    + C * v[i+1, n+1] 
        RHS = - A * v[i-1, n]      + (2 - B) * v[i, n]      - C * v[i+1, n]      

        Remove the boundaries from v, and set the boundary conditions:

        At maturity V = S-K                  => for all i, V[i,j=0]  = max(S-K, 0)
        Deep OTM options V = 0               => for all j, V[i=0,j]  = 0
        Deep ITM options V = S - K exp(-rt)  => for all j, V[i=-1,j] = S_max - K exp(-rt)
    """
    #### maturity boundary
    V[0, :] = np.maximum(S - K, 0)
    #### Low-S boundary:
    V[:, 0] = 0.0
    #### High-S boundary:
    V[:, -1] = S[-1] - K * np.exp(-r * tau)

    V_interior = V[:, 1:-1]
    """
    At each step, we will have

    L v[1:-1, n+1] = R v[1:-1, n] + boundary conditions

    But since the same L, R and LU factorisation is used at each step for all strikes in log space, we pre-emptively declare it now outside the loop:
    L = tri_diag[ A        (2+B)        C]
    R = tri_diag[-A        (2-B)       -C]
    """
    L = diags(
        [A, 2 + B, C],
        offsets=[-1, 0, 1],
        shape=(n_x - 2, n_x - 2)
    ).tocsc()

    R = diags(
        [-A, 2 - B, -C],
        offsets=[-1, 0, 1],
        shape=(n_x - 2, n_x - 2)
    ).tocsc()

    # Same L every timestep, so factorise once
    LU = splu(L)
    for n in range(n_t - 1):
        rhs = R @ V_interior[n]
        # Add boundary contributions
        rhs[0] -= A * (V[n, 0] + V[n + 1, 0])
        rhs[-1] -= C * (V[n, -1] + V[n + 1, -1])
        # 1D solve
        V_interior[n + 1] = LU.solve(rhs)

    return (V, {'S': S, 'tau': tau}) if return_full_V else V[-1, :], {'S': S}

def pde_crank_nicolson_v2(
        K,
        S0,
        r,
        T,
        sigma,
        s_steps=100,
        t_steps=100,
        n_std=5,
):
    """
    Crank-Nicolson solver for the Black-Scholes PDE for ONE strike.

    v3 indicates it is identical to v2, except now we only store a single column vector v_n, and discard previous timesteps.

    Unlike the multi-strike version:
        V.shape = (n_t, n_x)

    rather than:
        V.shape = (n_t, n_x, n_k)
    """

    n_x = s_steps + 1
    n_t = t_steps + 1

    # Grid
    x, dx, tau, dt = build_grid(K, S0, r, T, sigma, s_steps, t_steps, n_std=5)
    S = np.exp(x)

    # ------------------------------------------------------------------
    # Crank-Nicolson coefficients
    # ------------------------------------------------------------------
    a = dt * (0.5 * sigma ** 2) / dx ** 2
    b = dt * (r - 0.5 * sigma ** 2) / (2 * dx)
    c = -r * dt

    A = -(a - b)
    B = 2 * a - c
    C = -(a + b)

    # ------------------------------------------------------------------
    # Boundary / initial conditions
    # ------------------------------------------------------------------

    # tau = 0 corresponds to maturity:
    # V(S, 0) = max(S - K, 0)
    maturity_boundary = np.maximum(S - K, 0)
    #V[0, :] = np.maximum(S - K, 0)

    # Low-S boundary:
    # V -> 0 as S -> 0
    s_boundary_lower = np.zeros(n_t)
    #V[:, 0] = 0.0

    # High-S boundary:
    # V ~ S - K exp(-r tau)
    s_boundary_upper = S[-1] - K * np.exp(-r * tau)
    #V[:, -1] = S[-1] - K * np.exp(-r * tau)

    # Interior view -- modifications here modify V directly
    #V_interior = V[:, 1:-1]



    # ------------------------------------------------------------------
    # Crank-Nicolson matrices
    # ------------------------------------------------------------------
    L = diags(
        [A, 2 + B, C],
        offsets=[-1, 0, 1],
        shape=(n_x - 2, n_x - 2)
    ).tocsc()

    R = diags(
        [-A, 2 - B, -C],
        offsets=[-1, 0, 1],
        shape=(n_x - 2, n_x - 2)
    ).tocsc()

    # Same L every timestep, so factorise once
    LU = splu(L)

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------

    V_ = maturity_boundary[1:-1]
    for n in range(n_t - 1):

        # Now a 1D matrix-vector multiply rather than matrix-matrix
        rhs = R @ V_

        # Add boundary contributions
        rhs[0] -= A * (s_boundary_lower[n] + s_boundary_lower[n+1])
        rhs[-1] -= C * (s_boundary_upper[n] + s_boundary_upper[n+1])

        # 1D solve
        V_ = LU.solve(rhs)

    grid = {
        "S": S[1:-1],
    }

    return V_, grid

def pde_crank_nicolson_v3(
        K,
        S0,
        r,
        T,
        sigma,
        s_steps=100,
        t_steps=100
):
    """
    Crank-Nicolson solver for the Black-Scholes PDE for ONE strike.

    v4 indicates this method has all the optimisations of v3, but in addition, we replace the LU solver with:
    from scipy.linalg.lapack import dgttrf, dgttrsscipy

    Unlike the multi-strike version:
        V.shape = (n_t, n_x)

    rather than:
        V.shape = (n_t, n_x, n_k)
    """

    n_x = s_steps + 1
    n_t = t_steps + 1
    x, dx, tau, dt = build_grid(S0, K, r, T, sigma, s_steps, t_steps, n_std=5)
    S = np.exp(x)


    a = dt * (0.5 * sigma ** 2) / dx ** 2
    b = dt * (r - 0.5 * sigma ** 2) / (2 * dx)
    c = -r * dt
    A = -(a - b)
    B = 2 * a - c
    C = -(a + b)



    maturity_boundary = np.maximum(S - K, 0)
    # s_boundary_lower = np.zeros(n_t)
    s_boundary_upper = S[-1] - K * np.exp(-r * tau)
    bc_up = C * (s_boundary_upper[:-1] + s_boundary_upper[1:])
    #V[:, -1] = S[-1] - K * np.exp(-r * tau)


    dl, d, du, du2, ipiv, _ = dgttrf(np.full(n_x-3, A), np.full(n_x-2, 2 + B), np.full(n_x-3, C))
    dR = 2.0 - B
    V_ = maturity_boundary[1:-1]
    for n in range(n_t - 1):
        rhs = dR * V_
        rhs[1:] -= A * V_[:-1]
        rhs[:-1] -= C * V_[1:]
        rhs[-1] -= bc_up[n]
        V_, _ = dgttrs(dl, d, du, du2, ipiv, rhs, overwrite_b=1)

    grid = {
        "S": S[1:-1],
    }

    return V_, grid

class EuropeanOption:
    def __init__(self, K: float, T: float, type: Literal["call", "put"]):
        self.K = K
        self.T = T
        self.style = "European"
        self.type = type

    def payoff(self, S):
        return np.maximum(S - self.K, 0) if self.type == "call" else np.maximum(S - self.K, 0)

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

    def price_CN(self, S0, r, sigma, s_steps=100, t_steps=500, version : Literal["v1", "v2", "v3"] = "v1"):

        solvers = {
            "v1": pde_crank_nicolson,
            "v2": pde_crank_nicolson_v2,
            "v3": pde_crank_nicolson_v3,
        }

        solver = solvers[version]

        V, grid = solver(self.K, S0, r, self.T, sigma, s_steps=s_steps, t_steps=t_steps)
        call = np.interp(np.log(S0), np.log(grid['S']), V)
        return self.price(call, S0, r)



if __name__ == "__main__":
    print("hi")
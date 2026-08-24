import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import scipy.stats
from typing import Literal
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from scipy.linalg.lapack import dgttrf, dgttrs
from numba import njit


def plot_option_surface_2D(
        V,
        domain,
        labels,
        axis_to_fix,
        idx_to_take,
        ):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    V_2d = np.take(V, idx_to_take, axis=axis_to_fix)

    fixed = domain[axis_to_fix][idx_to_take]
    fixed_label = labels[axis_to_fix]

    coords = [d for i, d in enumerate(domain) if i != axis_to_fix]
    labels = [l for i, l in enumerate(labels) if i != axis_to_fix]
    X, Y = np.meshgrid(coords[0], coords[1], indexing='ij')
    surf = ax.plot_surface(X, Y, V_2d, cmap='viridis')

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel("Option Price")
    ax.set_title(f"{fixed_label} = {fixed:.4f}")
    ax.view_init(elev=35, azim=135)
    plt.show()

def plot_option_surface(strikes, S, V_S0):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X, Y = np.meshgrid(S, strikes, indexing='ij')
    surf = ax.plot_surface(X, Y, V_S0, cmap='viridis')
    ax.set_xlabel('Strike')
    ax.set_ylabel('Spot Price')
    ax.set_zlabel('Option Price')
    ax.view_init(elev=25, azim=-120)
    # fig.colorbar(surf, shrink=0.6, aspect=12, label='Price')
    plt.show()

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
    np.random.seed(seed)
    W = np.sqrt(T) * random.normal(0, 1, size=n)
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
    np.random.seed(seed)
    dt = T / steps
    dW = np.sqrt(dt) * random.normal(0, 1, size=(n, steps))
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
    np.random.seed(seed)
    dt = T / steps
    dW = np.sqrt(dt) * random.normal(0, 1, size=(n, steps))
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
    np.random.seed(seed)
    dt = T / steps
    dW = np.sqrt(dt) * random.normal(0, 1, size=(n, steps))
    S = np.zeros((n, steps + 1))
    S[:, 0] = S0
    for t in range(steps):
        S[:, t + 1] = S[:, t] * (1 + r * dt + sigma * dW[:, t] + 0.5 * sigma ** 2 * (dW[:, t] ** 2 - dt))
    return S

def strong_error(fn, S0, r, T, sigma, n, steps, seed):
    S_T = fn(S0, r, T, sigma, n=n, steps=steps, seed=seed)[:, -1]
    S_exact = gbm_exact_integration_reconstruct(S0, r, T, sigma, n=n, steps=steps, seed=seed)
    return np.abs(S_T - S_exact).mean()


def pde_crank_nicolson(
        strikes : float | np.ndarray,
        S0,
        r,
        T,
        sigma,
        s_steps=100,
        t_steps=100,
        return_full_V = False,
):
    """
    the crank nicolson method to solve the Black Scholes PDE for a range of strikes, and a plot of the option surface
    we allow for multiple strikes
    """

    """
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
    if isinstance(strikes, float | int) : strikes = np.atleast_1d(strikes).astype(float)

    n_x = s_steps + 1
    n_t = t_steps + 1
    n_k = len(strikes)
    strikes = np.sort(strikes)

    low = min(S0, strikes.min())
    high = max(S0, strikes.max())
    x_min = np.log(low) - 5 * sigma * np.sqrt(T)
    x_max = np.log(high) + 5 * sigma * np.sqrt(T)
    x, dx = np.linspace(x_min, x_max, n_x, retstep=True)
    tau, dt = np.linspace(0, T, n_t, retstep=True)
    S = np.exp(x)
    D = np.exp(-r * tau.reshape(-1, 1))

    """
    Define V
    #### Important:
    #### typically one writes v(x_i,t_n) = v[i, n]. But in numpy, it's better to have v(t_n,x_i) since we move forward in t
    """

    V = np.zeros((n_t, n_x, n_k))

    """
    Define:
    n = 1, 2, ..., t_steps + 1             [ index for tau        ]
    i = 1, 2, ..., S_steps + 1             [ index for step count ]
    Crank Nicolson method in log space:
    
    The various terms in their differenced forms (implicit - explicit for x coordinate, implicit in time):
    
    dv/dtau   = v[i, n+1] - v[i, n] / dtau                                                                                
    d^2v/dx^2 = [ (1/2)(v[i+1, n] - 2v[i, n]+ v[i-1, n])   + (1/2)(v[i+1, n+1] - 2v[i, n+1]+ v[i-1, n+1]) ] / dx^2   
    dv/dx  =    [ (1/2)(v[i+1, n]- v[i-1,n])               + (1/2)(v[i+1, n+1]- v[i-1, n+1])              ] / 2dx  
    v =         [ (1/2)(v[i, n])                           + (1/2)(v[i, n+1])                             ]

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
    c = dt * (-r)

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

    V_interior =  V[:, 1:-1, :]

    upper_S_boundary = S[-1] - D * strikes[None, :]
    lower_S_boundary = np.zeros((n_t, n_k))
    maturity_boundary = np.maximum(S[:, None] - strikes[None, :], 0)
    V[:,  0,  :] = lower_S_boundary
    V[:, -1,  :] = upper_S_boundary
    V[0,  :,  :] = maturity_boundary

    """
    At each step, we will have
    
    L v[1:-1, n+1] = R v[1:-1, n] + boundary conditions
    
    But since the same L, R and LU factorisation is used at each step for all strikes in log space, we pre-emptively declare it now outside the loop:
    L = tri_diag[ A        (2+B)        C]
    R = tri_diag[-A        (2-B)       -C]
    """
    L = diags([A, (2 + B), C], shape=(n_x-2, n_x-2), offsets=[-1, 0, 1]).tocsc()
    R = diags([-A, (2 - B), -C], shape=(n_x-2, n_x-2), offsets=[-1, 0, 1]).tocsc()
    LU = splu(L)
    """
    Want to plot the entire mesh later. But can drop old t-step terms for benchmarking
    """


    for n in range(n_t - 1):
        rhs = R @ V_interior[n, :, :]
        # add boundaries back
        rhs[0, :] -= A * (V[n, 0, :] + V[n + 1, 0, :])
        rhs[-1, :] -= C * (V[n, -1,  :] + V[n + 1, -1,  :])
        # solve for time step n+1
        V_interior[n + 1, :, :] = LU.solve(rhs)

    """
    maturity = np.linspace(0, T, n_t)
    plot_option_surface_2D(
        V,
        (S, maturity, strikes),
        ("spot", "maturity", "strike"),
        0,
        250,
    )
    """

    # only one strike
    if len(strikes) == 1: return (V[:,:,0], {'S': S, 'tau': tau}) if return_full_V else (V[-1,:,0], {'S': S})

    # multiple strikes
    if not return_full_V: return V[-1, :, :], {'S': S, 'K': strikes}
    return V, {'S': S, 'tau': tau, 'K': strikes}


def pde_crank_nicolson_v2(
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
    Crank-Nicolson solver for the Black-Scholes PDE

    v2 indicates this method is identical to v1, except it only works for ONE strike.

    Unlike the multi-strike version:
        V.shape = (n_t, n_x)
    rather than:
        V.shape = (n_t, n_x, n_k)
    """

    n_x = s_steps + 1
    n_t = t_steps + 1

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------
    low = min(S0, K)
    high = max(S0, K)

    x_min = np.log(low) - 5 * sigma * np.sqrt(T)
    x_max = np.log(high) + 5 * sigma * np.sqrt(T)

    x, dx = np.linspace(x_min, x_max, n_x, retstep=True)
    tau, dt = np.linspace(0, T, n_t, retstep=True)

    S = np.exp(x)

    # ------------------------------------------------------------------
    # Option value grid
    #
    # V[n, i] = V(tau_n, x_i)
    # ------------------------------------------------------------------
    V = np.zeros((n_t, n_x))

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
    V[0, :] = np.maximum(S - K, 0)

    # Low-S boundary:
    # V -> 0 as S -> 0
    V[:, 0] = 0.0

    # High-S boundary:
    # V ~ S - K exp(-r tau)
    V[:, -1] = S[-1] - K * np.exp(-r * tau)

    # Interior view -- modifications here modify V directly
    V_interior = V[:, 1:-1]

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
    for n in range(n_t - 1):

        # Now a 1D matrix-vector multiply rather than matrix-matrix
        rhs = R @ V_interior[n]

        # Add boundary contributions
        rhs[0] -= A * (V[n, 0] + V[n + 1, 0])
        rhs[-1] -= C * (V[n, -1] + V[n + 1, -1])

        # 1D solve
        V_interior[n + 1] = LU.solve(rhs)

    grid = {
        "S": S,
        "tau": tau,
    }

    return (V, {'S': S, 'tau': tau}) if return_full_V else V[-1, :], {'S': S}

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

    v3 indicates it is identical to v2, except now we only store a single column vector v_n, and discard previous timesteps.

    Unlike the multi-strike version:
        V.shape = (n_t, n_x)

    rather than:
        V.shape = (n_t, n_x, n_k)
    """

    n_x = s_steps + 1
    n_t = t_steps + 1

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------
    low = min(S0, K)
    high = max(S0, K)

    x_min = np.log(low) - 5 * sigma * np.sqrt(T)
    x_max = np.log(high) + 5 * sigma * np.sqrt(T)

    x, dx = np.linspace(x_min, x_max, n_x, retstep=True)
    tau, dt = np.linspace(0, T, n_t, retstep=True)

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

def pde_crank_nicolson_v4(
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
    low = min(S0, K)
    high = max(S0, K)
    x_min = np.log(low) - 5 * sigma * np.sqrt(T)
    x_max = np.log(high) + 5 * sigma * np.sqrt(T)
    x, dx = np.linspace(x_min, x_max, n_x, retstep=True)
    tau, dt = np.linspace(0, T, n_t, retstep=True)
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

    def price_CN(self, S0, r, sigma, s_steps=100, t_steps=500, version : Literal["v1", "v2", "v3", "v4"] = "v1"):

        solvers = {
            "v1": pde_crank_nicolson,
            "v2": pde_crank_nicolson_v2,
            "v3": pde_crank_nicolson_v3,
            "v4": pde_crank_nicolson_v4,
        }

        solver = solvers[version]

        V, grid = solver(self.K, S0, r, self.T, sigma, s_steps=s_steps, t_steps=t_steps)
        call = np.interp(np.log(S0), np.log(grid['S']), V)
        return self.price(call, S0, r)



if __name__ == "__main__":
    print("hi")
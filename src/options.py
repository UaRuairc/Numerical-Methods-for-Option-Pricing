import numpy as np
import scipy.stats
from typing import Literal
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from scipy.linalg.lapack import dgttrf, dgttrs
from numba import njit

## CN DERIVATION
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

    define a = dtau * (1/2 sigma^2)   / dx^2  = sigma^2         * (dtau/2dx^2) (This is the usual lambda Fourier number)
           b = dtau * (r-1/2 sigma^2) / 2 dx  = (r-1/2 sigma^2) * (dtau/2dx)
           c = dtau * (-r)

    
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

    LHS =   A * v[i-1, n+1]    + (2 + B) * v[i, n+1]    + C * v[i+1, n+1] 
    RHS = - A * v[i-1, n]      + (2 - B) * v[i, n]      - C * v[i+1, n]      

    Remove the boundaries from v, and set the boundary conditions:
    
    At maturity V = S-K                  => for all i, V[i,j=0]  = max(S-K, 0)
    Deep OTM options V = 0               => for all j, V[i=0,j]  = 0
    Deep ITM options V = S - K exp(-rt)  => for all j, V[i=-1,j] = S_max - K exp(-rt)
    
    At each step, we will have

    L v[1:-1, n+1] = R v[1:-1, n] + boundary conditions
    
    L = tri_diag[ A        (2+B)        C]
    R = tri_diag[-A        (2-B)       -C]
    
"""

## euler implicit derivation
"""
    Use implicit euler instead

    dv/dtau   = unchanged = (  v[i, n+1] - v[i, n]                     ) / dtau
    d^2v/dx^2 = changed   = (  v[i+1, n+1] - 2v[i, n+1] + v[i-1, n+1]  ) / dx^2
    dv/dx     = changed   = (  v[i+1, n+1] - v[i-1, n+1]               ) / 2dx
    v         = changed   =    v[i, n+1]

    dv/dtau = 1/2 sigma^2 d^2v/dx^2 + (r-1/2 sigma^2)dv/dx - rv

    substituting in (note dt is now just the spacing dt=dtau)

    a = dt * (0.5 * sigma ** 2) / dx ** 2
    b = dt * (r - 0.5 * sigma ** 2) / (2 * dx)
    c = -r * dt

    v[i, n+1] - v[i, n] =   a * (v[i+1, n+1] - 2v[i, n+1] + v[i-1, n+1])
                          + b * (v[i+1, n+1] - v[i-1, n+1])
                          + c * v[i, n+1]

    =>

    - (a - b) v[i-1, n+1]  +  (2a - c + 1) v[i, n+1]    - (a + b) v[i+1, n+1]    =     v[i, n]

    recall
            A = - (a-b)
            B = 2a - c
            C = - (a+b)

    LHS = A v[i-1, n+1]  +  (B + 1) v[i, n+1]    + C v[i+1, n+1]
    RHS = v[i, n]

    need two dampening steps,

    so dt -> dt/2
    => A, B, C -> A/2, B/2, C/2



    step 1:
        LHS = A v[i-1, n+1/2]  +  (B + 2) v[i, n+1/2]    + C v[i+1, n+1/2]
        RHS = 2 v[i, n]


    step 2:
        LHS = A v[i-1, n+1]    +  (B + 2) v[i, n+1]      + C v[i+1, n+1]
        RHS = 2 v[i, n+1/2]


"""
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

def cn_step(V, n, y, dl, d, du, du2, ipiv, bc_up):
    """advance V one Crank-Nicolson step in place."""
    np.multiply(V, 4.0, out=y)
    y[-1] -= bc_up[n]
    z, _ = dgttrs(dl, d, du, du2, ipiv, y, overwrite_b=1)
    np.subtract(z, V, out=V)


def rannacher_step(V, n, y, dl, d, du, du2, ipiv, bc_up_euler_implicit):
    """advance V one ranncher step (two euler implicit steps) in place."""

    ###
    # solve V[n+1] = 2 * L^-1 V[n] twice

    # step 1
    n_euler = 2 * n
    np.multiply(V, 2.0, out=y)
    y[-1] -= bc_up_euler_implicit[n_euler]
    z, _ = dgttrs(dl, d, du, du2, ipiv, y, overwrite_b=1)
    np.copyto(V, z)

    # step 2
    n_euler += 1
    np.multiply(V, 2.0, out=y)
    y[-1] -= bc_up_euler_implicit[n_euler]
    z, _ = dgttrs(dl, d, du, du2, ipiv, y, overwrite_b=1)
    np.copyto(V, z)

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
    # symmetric about ATM forward spot (in log space)
    centre = logK - r * T
    width = n_std * vol + 0.5 * sigma ** 2 * T

    # eventually want dx such that we are centred are logK - r*T
    low  = logK - (centre - width)
    high = (centre + width) - logK
    # guard against large maturity T (and large sigma or r)
    low = max(0.7 * n_std * vol, low)
    high = max(0.7 * n_std * vol, high)

    # make sure S0 is inside.
    low = max(low, logK - logS0)
    high = max(high, logS0 - logK)

    # need the number of steps is a whole number.
    # want (n_low + n_high) = s_steps
    # add padding so we can round n_low down

    padding_fraction = max(0.02,  2.0 / s_steps)
    extra_padding = (low+high) * padding_fraction

    low = low + extra_padding
    high = high + extra_padding
    dx = (low + high) / s_steps
    n_low = int(np.ceil(low/dx))
    n_low = max(n_low,2)
    n_low = min(n_low, s_steps-2)
    n_high = s_steps - n_low
    x = logK + dx * np.arange(-n_low, n_high+1)
    tau, dt = np.linspace(0, T, t_steps + 1, retstep=True)

    assert x.size == s_steps + 1, "node count must be s_steps + 1"
    if not (x[0] < logS0 < x[-1]):
        raise ValueError(
            f"grid does not bracket log(S0): {x[0]} < {logS0} < {x[-1]} is false "
            f"(K={K}, S0={S0}, T={T}, sigma={sigma})"
        )
    #print(f"S grid for: {K, S0, r, T, sigma, s_steps, t_steps, n_std = }")
    #print(f"{np.exp(x)[0]} < S < {np.exp(x)[-1]}")
    #Fourier_number = sigma**2 * dt / (2 * dx**2)
    #print(Fourier_number)

    return x, dx, tau, dt, n_low

def matrix_coefficients(dx, dt, r, sigma):


    a = dt * (0.5 * sigma ** 2) / dx ** 2
    b = dt * (r - 0.5 * sigma ** 2) / (2 * dx)
    c = -r * dt

    A = -(a - b)
    B = 2 * a - c
    C = -(a + b)

    matrix_coefficients = (A, B, C)

    return matrix_coefficients

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
    the crank nicolson method to solve the Black Scholes PDE
    """

    x, dx, tau, dt, _ = build_grid(K=K, S0=S0, r=r, T=T, sigma=sigma, s_steps=s_steps, t_steps=t_steps, n_std=5)
    A, B, C = matrix_coefficients(dx, dt, r, sigma)
    n_x, n_t = s_steps + 1, t_steps + 1

    S = np.exp(x)


    # ------------------------------------------------------------------
    # recall: typically one writes v(x_i,t_n) = v[i, n] but for speed in numpy it's better to have v(t_n,x_i) since we move forward in t
    # ------------------------------------------------------------------
    V = np.zeros((n_t, n_x))
    #### maturity boundary
    V[0, :] = np.maximum(S - K, 0)
    #### High-S boundary:
    V[:, -1] = S[-1] - K * np.exp(-r * tau)

    L = diags(
        diagonals=[A, 2 + B, C],
        offsets=[-1, 0, 1],
        shape=(n_x - 2, n_x - 2)
    ).tocsc()

    R = diags(
        diagonals=[-A, 2 - B, -C],
        offsets=[-1, 0, 1],
        shape=(n_x - 2, n_x - 2)
    ).tocsc()

    # Same L every timestep, so factorise once
    LU = splu(L)
    V_interior = V[:, 1:-1]
    for n in range(n_t - 1):
        rhs = R @ V_interior[n]
        # Add boundary contributions
        rhs[0] -= A * (V[n, 0] + V[n + 1, 0])
        rhs[-1] -= C * (V[n, -1] + V[n + 1, -1])
        # 1D solve
        V_interior[n + 1] = LU.solve(rhs)

    if return_full_V:
        return V, {'S': S, 'tau': tau}
    return V[-1, :], {'S': S}

def pde_crank_nicolson_v2(
        K,
        S0,
        r,
        T,
        sigma,
        s_steps=100,
        t_steps=100,
):
    """
    v2 indicates it is identical to v1, except now only allow one strike, and we only store a single column vector v_n, and discard previous timesteps.
    """
    x, dx, tau, dt, _ = build_grid(K=K, S0=S0, r=r, T=T, sigma=sigma, s_steps=s_steps, t_steps=t_steps, n_std=5)
    A, B, C = matrix_coefficients(dx, dt, r, sigma)
    n_x, n_t = s_steps + 1, t_steps + 1

    S = np.exp(x)


    # tau = 0 corresponds to maturity
    maturity_boundary = np.maximum(S - K, 0)
    # High-S boundary
    s_boundary_upper = S[-1] - K * np.exp(-r * tau)

    # L and R matrices
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
    V_ = maturity_boundary[1:-1]

    #######
    ## loop
    #######
    for n in range(n_t - 1):

        # Now a 1D matrix-vector multiply rather than matrix-matrix
        rhs = R @ V_

        # Add boundary contributions
        rhs[-1] -= C * (s_boundary_upper[n] + s_boundary_upper[n+1])

        # 1D solve
        V_ = LU.solve(rhs)

    V = np.empty(n_x)
    V[0] = 0.0
    V[1:-1] = V_
    V[-1] = s_boundary_upper[-1]

    grid = {"S": S}
    return V, grid

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
    v3 indicates this method has all the optimisations of v2, but in addition, we replace the LU solver with:
    from scipy.linalg.lapack import dgttrf, dgttrs
    """

    x, dx, tau, dt, _ = build_grid(K=K, S0=S0, r=r, T=T, sigma=sigma, s_steps=s_steps, t_steps=t_steps, n_std=5)
    A, B, C = matrix_coefficients(dx, dt, r, sigma)
    n_x, n_t = s_steps + 1, t_steps + 1

    S = np.exp(x)

    maturity_boundary = np.maximum(S - K, 0)
    s_boundary_upper = S[-1] - K * np.exp(-r * tau)
    bc_up = C * (s_boundary_upper[:-1] + s_boundary_upper[1:])

    # tridiagonal factorisation
    dl, d, du, du2, ipiv, _ = dgttrf(np.full(n_x-3, A), np.full(n_x-2, 2 + B), np.full(n_x-3, C))
    dR = 2.0 - B
    V_ = maturity_boundary[1:-1]
    for n in range(n_t - 1):
        rhs = dR * V_
        rhs[1:] -= A * V_[:-1]
        rhs[:-1] -= C * V_[1:]
        rhs[-1] -= bc_up[n]
        # tridiagonal solve
        V_, _ = dgttrs(dl, d, du, du2, ipiv, rhs, overwrite_b=1)

    V = np.empty(n_x)
    V[0] = 0.0
    V[1:-1] = V_
    V[-1] = s_boundary_upper[-1]

    grid = {
        "S": S,
    }

    return V, grid

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
    v4 indicates this method has all the optimisations of v3, but in addition, we optimise the cn loop further and add introduce rannacher smoother
    """

    x, dx, tau, dt, _ = build_grid(K=K, S0=S0, r=r, T=T, sigma=sigma, s_steps=s_steps, t_steps=t_steps, n_std=5)
    A, B, C = matrix_coefficients(dx, dt, r, sigma)
    n_x, n_t = s_steps + 1, t_steps + 1

    S = np.exp(x)

    s_boundary_upper = S[-1] - K * np.exp(-r * tau)
    bc_up = C * (s_boundary_upper[:-1] + s_boundary_upper[1:])

    do_rannacher_step = True
    n_rannacher = 1
    bc_up_euler_implicit=None

    if do_rannacher_step:
        tau_r = np.linspace(0.0, n_rannacher * dt, 2 * n_rannacher + 1)
        s_boundary_upper_r = S[-1] - K * np.exp(-r * tau_r)
        bc_up_euler_implicit = C * s_boundary_upper_r[1:]



    dl, d, du, du2, ipiv, info = dgttrf(np.full(n_x-3, A), np.full(n_x-2, 2 + B), np.full(n_x-3, C))
    if info != 0:
        raise RuntimeError(f"dgttrf failed: info={info}")

    ######
    # solve L V[n+1] = R V[n]  Eq. (***)

    # For euler implicit half_step R = 2I and Eq. (***) becomes:
    # V[n+1] = 2 * L^-1 (V[n] + boundary_conditions)

    # For CN step R = 4I - L and Eq. (***) becomes:
    # V[n+1] = (4 L^-1 - 1) (V[n] + boundary_conditions)

    V = np.empty(n_x)
    np.maximum(S - K, 0.0, out=V)
    V_interior = V[1:-1]
    y = np.empty_like(V_interior)

    for n in range(n_t - 1):

        if do_rannacher_step and n < n_rannacher:
            rannacher_step(V_interior, n, y, dl, d, du, du2, ipiv, bc_up_euler_implicit)
        else:
            cn_step(V_interior, n, y, dl, d, du, du2, ipiv, bc_up)

    V[0] = 0.0
    V[-1] = s_boundary_upper[-1]

    grid = {
        "S": S,
    }

    return V, grid

@njit
def tridiag_factor(lower, diag, upper):
    l = lower.copy()
    d = diag.copy()
    u = upper.copy()

    for i in range(d.size - 1):
        multiplier = l[i] / d[i]
        l[i] = multiplier
        d[i + 1] -= multiplier * u[i]

    return l, d, u

@njit
def tridiag_solve_inplace(l, d, u, x):
    n = d.size

    # Forward substitution
    for i in range(1, n):
        x[i] -= l[i - 1] * x[i - 1]

    # Back substitution
    x[n - 1] /= d[n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = (x[i] - u[i] * x[i + 1]) / d[i]

def cn_loop(
    V,
    work,
    l,
    d,
    u,
    bc_up,
    bc_up_euler_implicit,
    n_t,
    n_rannacher,
):
    for n in range(n_t - 1):

        if n < n_rannacher:

            # Rannacher step 1
            np.multiply(V, 2.0, out=work)
            work[-1] -= bc_up_euler_implicit[2 * n]

            tridiag_solve_inplace(l, d, u, work)

            np.copyto(V, work)

            # Rannacher step 2
            np.multiply(V, 2.0, out=work)
            work[-1] -= bc_up_euler_implicit[2 * n + 1]

            tridiag_solve_inplace(l, d, u, work)

            np.copyto(V, work)

        else:
            # Crank-Nicolson
            np.multiply(V, 4.0, out=work)
            work[-1] -= bc_up[n]

            tridiag_solve_inplace(l, d, u, work)

            np.subtract(work, V, out=V)

@njit
def _cn_loop(
    V,
    work,
    l,
    d,
    u,
    bc_up,
    bc_up_euler_implicit,
    n_t,
    n_rannacher,
):
    '''for a comparison between the cn loop that only JIT compiles tridiagonal solves'''
    for n in range(n_t - 1):

        # Rannacher smoothing
        if n < n_rannacher:

            # First implicit Euler half-step
            for i in range(V.size):
                work[i] = 2.0 * V[i]

            work[-1] -= bc_up_euler_implicit[2 * n]

            tridiag_solve_inplace(l, d, u, work)

            for i in range(V.size):
                V[i] = work[i]

            # Second implicit Euler half-step
            for i in range(V.size):
                work[i] = 2.0 * V[i]

            work[-1] -= bc_up_euler_implicit[2 * n + 1]

            tridiag_solve_inplace(l, d, u, work)

            for i in range(V.size):
                V[i] = work[i]

        else:
            # Crank-Nicolson
            for i in range(V.size):
                work[i] = 4.0 * V[i]

            work[-1] -= bc_up[n]

            tridiag_solve_inplace(l, d, u, work)

            for i in range(V.size):
                V[i] = work[i] - V[i]

def pde_crank_nicolson_v5(
        K,
        S0,
        r,
        T,
        sigma,
        s_steps=100,
        t_steps=100
):
    """
    v5 indicates this method has all the optimisations of v4, but we now just njit to compile a custom tridiagonal solver
    """

    x, dx, tau, dt, _ = build_grid(K=K, S0=S0, r=r, T=T, sigma=sigma, s_steps=s_steps, t_steps=t_steps, n_std=5)
    A, B, C = matrix_coefficients(dx, dt, r, sigma)
    n_x, n_t = s_steps + 1, t_steps + 1

    S = np.exp(x)


    s_boundary_upper = S[-1] - K * np.exp(-r * tau)
    bc_up = C * (s_boundary_upper[:-1] + s_boundary_upper[1:])

    do_rannacher_step = True
    n_rannacher = 1
    bc_up_euler_implicit=None

    if do_rannacher_step:
        tau_r = np.linspace(0.0, n_rannacher * dt, 2 * n_rannacher + 1)
        s_boundary_upper_r = S[-1] - K * np.exp(-r * tau_r)
        bc_up_euler_implicit = C * s_boundary_upper_r[1:]



    l, d, u = tridiag_factor(np.full(n_x-3, A), np.full(n_x-2, 2 + B), np.full(n_x-3, C))


    ######
    # solve L V[n+1] = R V[n]  Eq. (***)

    # For euler implicit half_step R = 2I and Eq. (***) becomes:
    # V[n+1] = 2 * L^-1 (V[n] + boundary_conditions)

    # For CN step R = 4I - L and Eq. (***) becomes:
    # V[n+1] = (4 L^-1 - 1) (V[n] + boundary_conditions)

    V = np.empty(n_x)
    np.maximum(S - K, 0.0, out=V)
    V_interior = V[1:-1]
    y = np.empty_like(V_interior)

    cn_loop(V_interior, y, l, d, u, bc_up, bc_up_euler_implicit, n_t, n_rannacher)

    V[0] = 0.0
    V[-1] = s_boundary_upper[-1]

    grid = {
        "S": S,
    }

    return V, grid

def pde_crank_nicolson_v6(
        K,
        S0,
        r,
        T,
        sigma,
        s_steps=100,
        t_steps=100
):
    """
    v6 indicates this method has all the optimisations of v5, but we now just njit the entire cn loop
    """

    x, dx, tau, dt, _ = build_grid(K=K, S0=S0, r=r, T=T, sigma=sigma, s_steps=s_steps, t_steps=t_steps, n_std=5)
    A, B, C = matrix_coefficients(dx, dt, r, sigma)
    n_x, n_t = s_steps + 1, t_steps + 1

    S = np.exp(x)




    s_boundary_upper = S[-1] - K * np.exp(-r * tau)
    bc_up = C * (s_boundary_upper[:-1] + s_boundary_upper[1:])

    do_rannacher_step = True
    n_rannacher = 1
    bc_up_euler_implicit=None

    if do_rannacher_step:
        tau_r = np.linspace(0.0, n_rannacher * dt, 2 * n_rannacher + 1)
        s_boundary_upper_r = S[-1] - K * np.exp(-r * tau_r)
        bc_up_euler_implicit = C * s_boundary_upper_r[1:]



    l, d, u = tridiag_factor(np.full(n_x-3, A), np.full(n_x-2, 2 + B), np.full(n_x-3, C))


    ######
    # solve L V[n+1] = R V[n]  Eq. (***)

    # For euler implicit half_step R = 2I and Eq. (***) becomes:
    # V[n+1] = 2 * L^-1 (V[n] + boundary_conditions)

    # For CN step R = 4I - L and Eq. (***) becomes:
    # V[n+1] = (4 L^-1 - 1) (V[n] + boundary_conditions)

    V = np.empty(n_x)
    np.maximum(S - K, 0.0, out=V)
    V_interior = V[1:-1]
    y = np.empty_like(V_interior)

    _cn_loop(V_interior, y, l, d, u, bc_up, bc_up_euler_implicit, n_t, n_rannacher)

    V[0] = 0.0
    V[-1] = s_boundary_upper[-1]

    grid = {
        "S": S,
    }

    return V, grid

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
            "v6": pde_crank_nicolson_v6      # modified v5 to JIT compile the whole cn loop, not just tridiagonal solve.
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
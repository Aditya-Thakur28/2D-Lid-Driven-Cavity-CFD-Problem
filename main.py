# ==========================================================
# 2D INCOMPRESSIBLE NAVIER-STOKES SOLVER
# Lid Driven Cavity — Vorticity-Streamfunction Formulation
# with Direct LU / Red-Black SOR Poisson Solver
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time
from numba import njit, prange


# ==========================================================
# SOLVER SELECTION
# ==========================================================

USE_DIRECT_SOLVER = True   # True = LU (fast, robust), False = Red-Black SOR


# ==========================================================
# DOMAIN PARAMETERS
# ==========================================================

Lx = 1
Ly = 1


# ==========================================================
# GRID GENERATION
# ==========================================================

nx =300
ny =300

print("Grid Size: " + str(nx) + " x " + str(ny))

dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
dx2 = dx ** 2
dy2 = dy ** 2


# ==========================================================
# PHYSICAL PARAMETERS
# ==========================================================

nu = float(input("Kinetic Viscosity: "))

U_lid = 1
Re = U_lid * Ly / nu
print("Reynolds Number:", Re)


# ==========================================================
# TIME STEPPING PARAMETERS
# ==========================================================

dt_max = 0.005
CFL = 0.5
nt = 200000


# ==========================================================
# STEADY-STATE CONVERGENCE
# ==========================================================

ss_tol = 1e-6


# ==========================================================
# RED-BLACK SOR PARAMETERS
# ==========================================================

sor_omega = 2.0 / (1.0 + np.sin(np.pi * dx))   # Optimal relaxation
sor_tol = 1e-6
sor_max_iter = 10000

if not USE_DIRECT_SOLVER:
    print(f"SOR relaxation factor: {sor_omega:.4f}")


# ==========================================================
# FIELD INITIALIZATION
# ==========================================================

omega = np.zeros((ny, nx))    # Vorticity
psi   = np.zeros((ny, nx))    # Streamfunction
u     = np.zeros((ny, nx))    # x-velocity (recovered from psi)
v     = np.zeros((ny, nx))    # y-velocity (recovered from psi)


# ==========================================================
# BUILD SPARSE LAPLACIAN FOR STREAMFUNCTION POISSON EQ.
# (Direct solver only — factorized once with LU)
# ==========================================================

if USE_DIRECT_SOLVER:
    print("Building streamfunction Laplacian matrix...")
    t_build = time.time()

    N_interior = (ny - 2) * (nx - 2)

    def idx(i, j):
        return (i - 1) * (nx - 2) + (j - 1)

    rows = []
    cols = []
    vals = []

    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            k = idx(i, j)
            center_coeff = -2.0 / dx2 - 2.0 / dy2

            # Right neighbor
            if j + 1 <= nx - 2:
                rows.append(k); cols.append(idx(i, j + 1)); vals.append(1.0 / dx2)
            # else: psi=0 on right wall → no contribution

            # Left neighbor
            if j - 1 >= 1:
                rows.append(k); cols.append(idx(i, j - 1)); vals.append(1.0 / dx2)

            # Top neighbor
            if i + 1 <= ny - 2:
                rows.append(k); cols.append(idx(i + 1, j)); vals.append(1.0 / dy2)

            # Bottom neighbor
            if i - 1 >= 1:
                rows.append(k); cols.append(idx(i - 1, j)); vals.append(1.0 / dy2)

            rows.append(k); cols.append(k); vals.append(center_coeff)

    A = sp.csc_matrix((vals, (rows, cols)), shape=(N_interior, N_interior))
    psi_lu = spla.splu(A)

    print(f"Laplacian built & factorized in {time.time() - t_build:.2f}s")


# ==========================================================
# RED-BLACK SOR POISSON SOLVER (NUMBA-PARALLELIZED)
# ==========================================================

@njit(parallel=True, cache=True)
def rb_sor_sweep(psi, rhs, nx, ny, dx2, dy2, omega_sor, color):
    """One Red-Black SOR sweep. color=0 for red, color=1 for black."""
    coeff = 1.0 / (-2.0 / dx2 - 2.0 / dy2)
    for i in prange(1, ny - 1):
        for j in range(1, nx - 1):
            if (i + j) % 2 == color:
                psi_gs = coeff * (
                    rhs[i, j]
                    - (psi[i, j + 1] + psi[i, j - 1]) / dx2
                    - (psi[i + 1, j] + psi[i - 1, j]) / dy2
                )
                psi[i, j] = (1.0 - omega_sor) * psi[i, j] + omega_sor * psi_gs


@njit(cache=True)
def rb_sor_residual(psi, rhs, nx, ny, dx2, dy2):
    """Compute max residual of the Poisson equation."""
    max_res = 0.0
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            lap = ((psi[i, j + 1] - 2.0 * psi[i, j] + psi[i, j - 1]) / dx2
                 + (psi[i + 1, j] - 2.0 * psi[i, j] + psi[i - 1, j]) / dy2)
            res = abs(lap - rhs[i, j])
            if res > max_res:
                max_res = res
    return max_res


# ==========================================================
# VORTICITY TRANSPORT RHS (2nd-ORDER UPWIND ADVECTION)
# ==========================================================

@njit(parallel=True, cache=True)
def compute_vorticity_rhs(omega, u, v, H, nx, ny, dx, dy, nu):
    """Compute H = -u·∇ω + ν·∇²ω using central differences."""
    for i in prange(1, ny - 1):
        for j in range(1, nx - 1):
            # Advection (central differences)
            dodx = (omega[i, j+1] - omega[i, j-1]) / (2.0 * dx)
            dody = (omega[i+1, j] - omega[i-1, j]) / (2.0 * dy)

            # Diffusion (central differences)
            d2odx2 = (omega[i,j+1] - 2.0*omega[i,j] + omega[i,j-1]) / (dx*dx)
            d2ody2 = (omega[i+1,j] - 2.0*omega[i,j] + omega[i-1,j]) / (dy*dy)

            H[i, j] = -(u[i,j]*dodx + v[i,j]*dody) + nu*(d2odx2 + d2ody2)

    return H


# ==========================================================
# WARM UP NUMBA
# ==========================================================

print("Compiling Numba kernels (one-time cost)...")
_d1 = np.zeros((ny, nx))
_d2 = np.zeros((ny, nx))
compute_vorticity_rhs(_d1, _d1, _d1, _d2, nx, ny, dx, dy, nu)
rb_sor_sweep(_d1.copy(), _d2, nx, ny, dx2, dy2, 1.5, 0)
rb_sor_sweep(_d1.copy(), _d2, nx, ny, dx2, dy2, 1.5, 1)
rb_sor_residual(_d1, _d2, nx, ny, dx2, dy2)
del _d1, _d2
print("Numba compilation done.")


# ==========================================================
# MAIN TIME INTEGRATION LOOP
# ==========================================================

print("\nStarting time integration...")
t_start = time.time()

H = np.zeros((ny, nx))   # Vorticity RHS

for n in range(nt):

    omega_old = omega.copy()


    # ==========================================================
    # STEP 1: WALL VORTICITY (Thom's formula)
    # ==========================================================

    # Bottom wall (i=0): u=0, v=0
    omega[0, :] = -2.0 * psi[1, :] / dy2

    # Top wall (i=ny-1): u=U_lid at interior, u=0 at corners
    omega[-1, :] = -2.0 * psi[-2, :] / dy2
    omega[-1, 1:-1] -= 2.0 * U_lid / dy

    # Left wall (j=0): u=0, v=0
    omega[1:-1, 0] = -2.0 * psi[1:-1, 1] / dx2

    # Right wall (j=nx-1): u=0, v=0
    omega[1:-1, -1] = -2.0 * psi[1:-1, -2] / dx2


    # ==========================================================
    # STEP 2: RECOVER VELOCITY FROM STREAMFUNCTION
    # ==========================================================

    u[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * dy)
    v[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2.0 * dx)

    # Boundary velocities
    u[0, :] = 0.0;      u[-1, :] = U_lid
    u[:, 0] = 0.0;      u[:, -1] = 0.0
    v[0, :] = 0.0;      v[-1, :] = 0.0
    v[:, 0] = 0.0;      v[:, -1] = 0.0

    # Corner override: no lid velocity at corners
    u[-1, 0] = 0.0;     u[-1, -1] = 0.0


    # ==========================================================
    # ADAPTIVE TIME STEP (CFL CONTROL)
    # ==========================================================

    umax = np.max(np.abs(u))
    vmax = np.max(np.abs(v))
    vel_max = max(umax, vmax, 1e-3)

    dt_conv = CFL * min(dx / vel_max, dy / vel_max)
    dt_diff = CFL / (2.0 * nu * (1.0 / dx2 + 1.0 / dy2))
    dt = min(dt_conv, dt_diff, dt_max)


    # ==========================================================
    # STEP 3: VORTICITY TRANSPORT (Forward Euler)
    # ==========================================================

    H[:] = 0.0
    compute_vorticity_rhs(omega, u, v, H, nx, ny, dx, dy, nu)

    omega[1:-1, 1:-1] += dt * H[1:-1, 1:-1]


    # ==========================================================
    # STEP 4: SOLVE POISSON FOR STREAMFUNCTION  ∇²ψ = −ω
    # ==========================================================

    rhs_psi = -omega

    if USE_DIRECT_SOLVER:
        psi_sol = psi_lu.solve(rhs_psi[1:-1, 1:-1].ravel())
        psi[1:-1, 1:-1] = psi_sol.reshape((ny - 2, nx - 2))
    else:
        # Red-Black SOR (checkerboard method)
        for sor_iter in range(sor_max_iter):
            rb_sor_sweep(psi, rhs_psi, nx, ny, dx2, dy2, sor_omega, 0)  # Red
            rb_sor_sweep(psi, rhs_psi, nx, ny, dx2, dy2, sor_omega, 1)  # Black

            if sor_iter % 20 == 0 and sor_iter > 0:
                sor_res = rb_sor_residual(psi, rhs_psi, nx, ny, dx2, dy2)
                if sor_res < sor_tol:
                    break

    # psi = 0 on all walls (never modified)


    # ==========================================================
    # CONVERGENCE CHECK
    # ==========================================================

    residual = np.max(np.abs(omega - omega_old))

    if n % 500 == 0:
        elapsed = time.time() - t_start
        print(f"Iteration {n:6d} | Residual: {residual:.2e} | dt: {dt:.6f} | Time: {elapsed:.1f}s")

    if residual < ss_tol and n > 10:
        print(f"\n=== Converged at iteration {n} | Residual: {residual:.2e} ===")
        break


total_time = time.time() - t_start
print(f"\nSimulation complete in {total_time:.1f}s ({total_time/60:.1f} min)")
print("Max u:", np.max(u))
print("Max v:", np.max(np.abs(v)))


# ==========================================================
# GRID FOR VISUALIZATION
# ==========================================================

x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)


# ==========================================================
# STREAMLINE PLOT
# ==========================================================

plt.figure(figsize=(6, 5))
plt.streamplot(X, Y, u, v)
plt.title("Velocity Field")
plt.show()


# ==========================================================
# GHIA ET AL. (1982) VALIDATION DATA — ALL REYNOLDS NUMBERS
# Tables I & II from: Ghia, Ghia & Shin (1982), J. Comp. Phys.
# ==========================================================

# --- y-locations for U-velocity along vertical centerline (Table I) ---
ghia_y = np.array([
    1.0000, 0.9766, 0.9688, 0.9609, 0.9531,
    0.8516, 0.7344, 0.6172, 0.5000, 0.4531,
    0.2813, 0.1719, 0.1016, 0.0703, 0.0625,
    0.0547, 0.0000
])

# --- U-velocity data for each Re ---
ghia_u = {
    100: np.array([
         1.00000,  0.84123,  0.78871,  0.73722,  0.68717,
         0.23151,  0.00332, -0.13641, -0.20581, -0.21090,
        -0.15662, -0.10150, -0.06434, -0.04775, -0.04192,
        -0.03717,  0.00000
    ]),
    400: np.array([
         1.00000,  0.75837,  0.68439,  0.61756,  0.55892,
         0.29093,  0.16256,  0.02135, -0.11477, -0.17119,
        -0.32726, -0.24299, -0.14612, -0.10338, -0.09266,
        -0.08186,  0.00000
    ]),
    1000: np.array([
         1.00000,  0.65928,  0.57492,  0.51117,  0.46604,
         0.33304,  0.18719,  0.05702, -0.06080, -0.10648,
        -0.27805, -0.38289, -0.29730, -0.22220, -0.20196,
        -0.18109,  0.00000
    ]),
    3200: np.array([
         1.00000,  0.53236,  0.48296,  0.46547,  0.46101,
         0.34682,  0.19791,  0.07156, -0.04272, -0.86636e-1,
        -0.24427, -0.34323, -0.41933, -0.37827, -0.35344,
        -0.32407,  0.00000
    ]),
    5000: np.array([
         1.00000,  0.48223,  0.46120,  0.45992,  0.46036,
         0.33556,  0.20087,  0.08183, -0.03039, -0.07404,
        -0.22855, -0.33050, -0.40435, -0.43643, -0.42901,
        -0.41165,  0.00000
    ]),
    7500: np.array([
         1.00000,  0.47244,  0.47048,  0.47323,  0.47167,
         0.34228,  0.20591,  0.08342, -0.03800, -0.07503,
        -0.23176, -0.32393, -0.38324, -0.43025, -0.43590,
        -0.43154,  0.00000
    ]),
    10000: np.array([
         1.00000,  0.47221,  0.47783,  0.48070,  0.47804,
         0.34635,  0.20673,  0.08344,  0.03111, -0.07540,
        -0.23186, -0.32709, -0.38000, -0.41657, -0.42537,
        -0.42735,  0.00000
    ]),
}

# --- x-locations for V-velocity along horizontal centerline (Table II) ---
ghia_x = np.array([
    1.0000, 0.9688, 0.9609, 0.9531, 0.9453,
    0.9063, 0.8594, 0.8047, 0.5000, 0.2344,
    0.2266, 0.1563, 0.0938, 0.0625, 0.0469,
    0.0391, 0.0313, 0.0000
])

# --- V-velocity data for each Re ---
ghia_v = {
    100: np.array([
         0.00000, -0.05906, -0.07391, -0.08864, -0.10313,
        -0.16914, -0.22445, -0.24533,  0.05454,  0.17527,
         0.17507,  0.16077,  0.12317,  0.09197,  0.07604,
         0.06647,  0.05600,  0.00000
    ]),
    400: np.array([
         0.00000, -0.12146, -0.15663, -0.19254, -0.22847,
        -0.23827, -0.44993, -0.38598,  0.05186,  0.30174,
         0.30203,  0.28124,  0.22965,  0.18360,  0.15852,
         0.14515,  0.12935,  0.00000
    ]),
    1000: np.array([
         0.00000, -0.21388, -0.27669, -0.33714, -0.39188,
        -0.51550, -0.42665, -0.31966,  0.02526,  0.32235,
         0.33075,  0.37095,  0.32627,  0.26154,  0.22279,
         0.20196,  0.17860,  0.00000
    ]),
    3200: np.array([
         0.00000, -0.39017, -0.47425, -0.52357, -0.54053,
        -0.44307, -0.37401, -0.31184,  0.00999,  0.28188,
         0.29030,  0.37119,  0.42768,  0.37095,  0.32627,
         0.30174,  0.27280,  0.00000
    ]),
    5000: np.array([
         0.00000, -0.49774, -0.55069, -0.55408, -0.52876,
        -0.41442, -0.36214, -0.30018,  0.00945,  0.27280,
         0.28066,  0.35368,  0.42951,  0.41165,  0.37095,
         0.34682,  0.31966,  0.00000
    ]),
    7500: np.array([
         0.00000, -0.53858, -0.55216, -0.52347, -0.48590,
        -0.41050, -0.36213, -0.30448,  0.00824,  0.27348,
         0.28117,  0.35060,  0.41824,  0.43154,  0.39188,
         0.36413,  0.33484,  0.00000
    ]),
    10000: np.array([
         0.00000, -0.54302, -0.52987, -0.49099, -0.45863,
        -0.41496, -0.36737, -0.30719,  0.00831,  0.27224,
         0.28003,  0.35070,  0.41487,  0.43590,  0.40187,
         0.37401,  0.34323,  0.00000
    ]),
}

# --- Find the closest matching Re ---
ghia_re_values = sorted(ghia_u.keys())
closest_re = min(ghia_re_values, key=lambda r: abs(r - Re))
re_match = abs(closest_re - Re) < 1

if re_match:
    ghia_u_data = ghia_u[closest_re]
    ghia_v_data = ghia_v[closest_re]
    print(f"\nUsing Ghia et al. (1982) data for Re = {closest_re}")
else:
    ghia_u_data = ghia_u[100]
    ghia_v_data = ghia_v[100]
    print(f"\nNo exact Ghia data for Re = {Re:.0f}. Showing Re=100 for reference.")
    print(f"Available Re values: {ghia_re_values}")


# ==========================================================
# CENTERLINE VALIDATION PLOTS
# ==========================================================

center_x = nx // 2
center_y = ny // 2

u_centerline = u[:, center_x]
v_centerline = v[center_y, :]

y_vals = np.linspace(0, Ly, ny)
x_vals = np.linspace(0, Lx, nx)

re_label = f"Ghia et al. Re={closest_re}" if re_match else "Ghia et al. Re=100 (ref)"

plt.figure()
plt.plot(u_centerline, y_vals, 'b-', label='Simulation', linewidth=1.5)
plt.plot(ghia_u_data, ghia_y, 'ro', label=re_label, markersize=6)
plt.title(f"U velocity along vertical centerline (x=0.5) — Re={Re:.0f}")
plt.xlabel("u")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(x_vals, v_centerline, 'b-', label='Simulation', linewidth=1.5)
plt.plot(ghia_x, ghia_v_data, 'ro', label=re_label, markersize=6)
plt.title(f"V velocity along horizontal centerline (y=0.5) — Re={Re:.0f}")
plt.xlabel("x")
plt.ylabel("v")
plt.legend()
plt.grid()
plt.show()


# ==========================================================
# QUANTITATIVE ERROR ANALYSIS
# ==========================================================

if re_match:
    u_interp = np.interp(ghia_y, y_vals, u_centerline)
    v_interp = np.interp(ghia_x, x_vals, v_centerline)

    u_errors = np.abs(u_interp[1:-1] - ghia_u_data[1:-1])
    v_errors = np.abs(v_interp[1:-1] - ghia_v_data[1:-1])

    u_max_pct = np.max(u_errors) / U_lid * 100
    v_max_pct = np.max(v_errors) / U_lid * 100
    u_avg_pct = np.mean(u_errors) / U_lid * 100
    v_avg_pct = np.mean(v_errors) / U_lid * 100

    print("\n" + "=" * 50)
    print(f"VALIDATION AGAINST GHIA ET AL. (1982) — Re = {closest_re}")
    print("=" * 50)
    print(f"U-velocity: Max error = {u_max_pct:.2f}%  |  Avg error = {u_avg_pct:.2f}%")
    print(f"V-velocity: Max error = {v_max_pct:.2f}%  |  Avg error = {v_avg_pct:.2f}%")
    print("=" * 50)


# ==========================================================
# VELOCITY MAGNITUDE CONTOUR
# ==========================================================

velocity = np.sqrt(u**2 + v**2)

plt.figure(figsize=(6, 5))
plt.contourf(X, Y, velocity, 50)
plt.colorbar(label="Velocity Magnitude")
plt.title("Velocity Magnitude Contour")
plt.show()


# ==========================================================
# VORTICITY PLOT
# ==========================================================

plt.figure(figsize=(6, 5))
plt.contourf(X, Y, omega, 50)
plt.colorbar(label="Vorticity")
plt.title("Vorticity Field")
plt.show()

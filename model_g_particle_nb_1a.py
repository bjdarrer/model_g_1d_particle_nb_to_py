"""
Translated Model G "particle.nb" -> Python
Author: Assistant (translated for Brendan)
Notes:
- This reproduces the 1D simulation in the Mathematica notebook (dimension==1).
- Method of lines: second-order central differences for Laplacian in x, Dirichlet BC at x=+-L/2.
- Time integration uses scipy.integrate.solve_ivp (RK23 by default). A TensorFlow RK4 integrator is provided as an alternative (explicit), but must use small dt for stability.
- Exports final plots and an MP4 animation (requires ffmpeg/imageio).

Run: python particle_model_g.py

"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import imageio
import os

# ---------- Parameters taken from the Mathematica notebook (eqs17) ----------
params = {
    'a': 14.0,
    'b': 29.0,
    'dx': 1.0,
    'dy': 12.0,
    'p': 1.0,
    'q': 1.0,
    'g': 1.0/10.0,
    's': 0.0,
    'u': 0.0,
    'v': 0.0,   # scalar for 1D (v is a vector in Mathematica), here v=0
    'w': 0.0
}

# Domain and discretization (matches L=100, T=30 in notebook)
L = 100.0
Tfinal = 30.0
nx = 801                 # number of grid points (odd -> center at index mid)
xgrid = np.linspace(-L/2, L/2, nx)
dx_space = xgrid[1] - xgrid[0]

# Seeds and forcing (chi). The notebook defines a bell function
def bell(s, x):
    return np.exp(- (x/s)**2 / 2.0)

# seed count: notebook default nseeds=1. We'll implement nseeds options 1,2,3
nseeds = 1
Tseed = 10.0

def chi(x, t, nseeds=nseeds):
    if nseeds == 1:
        return -bell(1.0, x) * bell(3.0, t - Tseed)
    elif nseeds == 2:
        return -(bell(1.0, x + 3.303/2) + bell(1.0, x - 3.303/2)) * bell(3.0, t - Tseed)
    else:
        return -(bell(1.0, x + 3.314) + bell(1.0, x) + bell(1.0, x - 3.314)) * bell(3.0, t - Tseed)

# Homogeneous steady-state values (G0, X0, Y0) from notebook
a = params['a']; b = params['b']; p = params['p']; q = params['q']; g = params['g']; s = params['s']; u = params['u']; w = params['w']

G0 = (a + g*w)/(q - g*p)
X0 = (p*a + q*w)/(q - g*p)
# Note: when u==0 and s==0 the formula simplifies
Y0 = (s*X0**2 + b)*X0/(X0**2 + u) if (X0**2 + u)!=0 else 0.0

print('G0, X0, Y0 =', G0, X0, Y0)

# Unknowns: pG, pX, pY (perturbations). Equations eqs13 in notebook.
# For 1D, Laplacian -> d2/dx2, grad -> d/dx. v dot grad = v * d/dx where v scalar.

# Build second derivative operator with Dirichlet BC (zero at boundaries)
def laplacian_1d(u_arr):
    # u_arr is 1D array of length nx
    # second-order central differences for interior points, Dirichlet zero at boundaries
    dudxx = np.zeros_like(u_arr)
    # interior 1..nx-2
    dudxx[1:-1] = (u_arr[2:] - 2*u_arr[1:-1] + u_arr[:-2]) / (dx_space**2)
    # boundaries remain 0 (Dirichlet as in notebook pG(L/2,t)==0 etc.)
    return dudxx

# Right-hand side for method-of-lines system: flatten [pG, pX, pY] concatenated
def rhs(t, y_flat):
    # reshape
    N = nx
    pG = y_flat[0:N]
    pX = y_flat[N:2*N]
    pY = y_flat[2*N:3*N]

    # spatial derivatives
    lap_pG = laplacian_1d(pG)
    lap_pX = laplacian_1d(pX)
    lap_pY = laplacian_1d(pY)

    # gradient first derivative (central differences), with dirichlet boundaries -> 0 derivative at boundary ignored
    dpGdx = np.zeros_like(pG)
    dpXdx = np.zeros_like(pX)
    dpYdx = np.zeros_like(pY)
    dpGdx[1:-1] = (pG[2:] - pG[:-2]) / (2*dx_space)
    dpXdx[1:-1] = (pX[2:] - pX[:-2]) / (2*dx_space)
    dpYdx[1:-1] = (pY[2:] - pY[:-2]) / (2*dx_space)

    # parameters
    dxp = params['dx']
    dyp = params['dy']
    b = params['b']
    p_par = params['p']
    q_par = params['q']
    g_par = params['g']
    s_par = params['s']
    u_par = params['u']
    v_par = params['v']
    w_par = params['w']

    # Forcing chi evaluated at each grid point (only added to pX equation per notebook)
    chi_vec = chi(xgrid, t)

    # eqs13 (note signs and ordering) from notebook
    dpGdt = lap_pG - v_par * dpGdx - q_par * pG + g_par * pX

    # Nonlinear terms: s( (pX+X0)^3 - X0^3 ) and ((pX+X0)^2 (pY+Y0) - X0^2 Y0)
    Xtot = pX + X0
    Ytot = pY + Y0
    nonlinear_s = s_par * (Xtot**3 - X0**3)  # equals s*( (pX+X0)^3 - X0^3 )
    nonlinear_xy = (Xtot**2 * Ytot - X0**2 * Y0)

    dpXdt = dxp * lap_pX - v_par * dpXdx + p_par * pG - (1.0 + b) * pX + u_par * pY - nonlinear_s + nonlinear_xy + chi_vec

    dpYdt = dyp * lap_pY - v_par * dpYdx + b * pX - u_par * pY + ( - nonlinear_xy + nonlinear_s )

    # enforce Dirichlet BCs: set time-derivative at boundary so value stays zero
    dpGdt[0] = 0.0; dpGdt[-1] = 0.0
    dpXdt[0] = 0.0; dpXdt[-1] = 0.0
    dpYdt[0] = 0.0; dpYdt[-1] = 0.0

    return np.concatenate([dpGdt, dpXdt, dpYdt])

# initial condition: zeros (noted in notebook pG[x,0]==0 etc.)
y0 = np.zeros(3*nx)

# time span and options: notebook used MaxStepSize ~ 0.03, T=30
t_span = (0.0, Tfinal)
max_step = 0.03

print('Starting solve_ivp... (this may take a minute)')
#sol = solve_ivp(rhs, t_span, y0, method='RK23', max_step=max_step, atol=1e-6, rtol=1e-6)

sol = solve_ivp(rhs, t_span, y0, method='RK23', max_step=max_step,
                atol=1e-6, rtol=1e-6, dense_output=True)

print('solve_ivp finished: status', sol.status, 'message:', sol.message)

# Extract solution at requested timepoints
# We'll sample the solution at 200 evenly spaced times for animation
nt_anim = 200
ts = np.linspace(0, Tfinal, nt_anim)
Ysol = sol.sol(ts)  # shape (3*nx, nt_anim)

# Helper to plot snapshot at time index i
def plot_snapshot(i, savepath=None):
    pG = Ysol[0:nx, i]
    pX = Ysol[nx:2*nx, i]
    pY = Ysol[2*nx:3*nx, i]

    plt.figure(figsize=(12,6))
    plt.plot(xgrid, pY, label='pY (Y)', linewidth=1.5)
    plt.plot(xgrid, pG, label='pG (G)')
    plt.plot(xgrid, pX/10.0, label='pX/10 (X scaled)')
    plt.title(f'Model G potentials t={ts[i]:.3f}')
    plt.xlabel('Space')
    plt.legend()
    plt.grid(True)
    if savepath:
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()

# Save final snapshot
os.makedirs('out_frames', exist_ok=True)
plot_snapshot(-1, savepath=os.path.join('out_frames', 'final.png'))
print('Final core amplitudes at x=0:')
# find index closest to x=0
mid_idx = np.argmin(np.abs(xgrid))
pG_final = Ysol[mid_idx, -1]
pX_final = Ysol[nx + mid_idx, -1]
pY_final = Ysol[2*nx + mid_idx, -1]
print('pY[0,T] =', pY_final)
print('pG[0,T] =', pG_final)
print('pX[0,T] =', pX_final)

# Build animation (MP4). We'll write PNG frames then assemble to MP4 using imageio
print('Rendering frames...')
filenames = []
for i in range(nt_anim):
    fname = os.path.join('out_frames', f'frame_{i:04d}.png')
    plot_snapshot(i, savepath=fname)
    filenames.append(fname)

print('Writing animation to particle_model_g.mp4 (may take a while)')
with imageio.get_writer('particle_model_g.mp4', fps=16) as writer:
    for fname in filenames:
        image = imageio.imread(fname)
        writer.append_data(image)

print('Done. Files written to current directory: out_frames/, particle_model_g.mp4')

# ---------------- Optional: Simple TensorFlow explicit RK4 integrator ----------------
# This is provided as an alternative path if you want to run on GPU. It is explicit and
# not memory-optimized for very large grids. Use small dt (e.g. 1e-3) for stability.
try:
    import tensorflow as tf
    tf_available = True
except Exception:
    tf_available = False

if tf_available:
    print('\nTensorFlow available. Providing optional TF integrator function tf_integrate_explicit_RK4().')
    def tf_integrate_explicit_RK4(nsteps=3000, dt=0.01):
        # Build initial tensors
        x_tf = tf.constant(xgrid, dtype=tf.float32)
        pG = tf.Variable(tf.zeros_like(x_tf), dtype=tf.float32)
        pX = tf.Variable(tf.zeros_like(x_tf), dtype=tf.float32)
        pY = tf.Variable(tf.zeros_like(x_tf), dtype=tf.float32)

        @tf.function
        def lap_tf(u):
            # assume Dirichlet BC (0 at boundaries)
            u_pad = tf.concat([[0.0], u[1:-1], [0.0]], axis=0)  # keep shape
            # central second derivative for interior
            u_shift_r = tf.concat([u[1:], [0.0]], axis=0)
            u_shift_l = tf.concat([[0.0], u[:-1]], axis=0)
            return (u_shift_r - 2.0*u + u_shift_l) / (dx_space**2)

        def rhs_tf(t, state):
            # state is concatenated vector of pG, pX, pY
            pG_s = state[0:nx]
            pX_s = state[nx:2*nx]
            pY_s = state[2*nx:3*nx]
            lapG = lap_tf(pG_s)
            lapX = lap_tf(pX_s)
            lapY = lap_tf(pY_s)
            # central diff for gradient
            dpGdx = tf.concat([[0.0], (pG_s[2:] - pG_s[:-2])/(2*dx_space), [0.0]], axis=0)
            dpXdx = tf.concat([[0.0], (pX_s[2:] - pX_s[:-2])/(2*dx_space), [0.0]], axis=0)
            dpYdx = tf.concat([[0.0], (pY_s[2:] - pY_s[:-2])/(2*dx_space), [0.0]], axis=0)

            chi_vec = tf.cast(chi(xgrid, t), tf.float32)

            dpGdt = lapG - v_par * dpGdx - q_par * pG_s + g_par * pX_s
            Xtot = pX_s + X0
            Ytot = pY_s + Y0
            nonlinear_s = s_par * (Xtot**3 - X0**3)
            nonlinear_xy = (Xtot**2 * Ytot - X0**2 * Y0)
            dpXdt = dxp * lapX - v_par * dpXdx + p_par * pG_s - (1.0 + b) * pX_s + u_par * pY_s - nonlinear_s + nonlinear_xy + chi_vec
            dpYdt = dyp * lapY - v_par * dpYdx + b * pX_s - u_par * pY_s + (-nonlinear_xy + nonlinear_s)

            # enforce BCs: keep boundary 0
            dpGdt = tf.concat([[0.0], dpGdt[1:-1], [0.0]], axis=0)
            dpXdt = tf.concat([[0.0], dpXdt[1:-1], [0.0]], axis=0)
            dpYdt = tf.concat([[0.0], dpYdt[1:-1], [0.0]], axis=0)

            return tf.concat([dpGdt, dpXdt, dpYdt], axis=0)

        # Initial state
        state = tf.zeros(3*nx, dtype=tf.float32)
        t = 0.0
        traj = []
        for k in range(nsteps):
            k1 = rhs_tf(t, state)
            k2 = rhs_tf(t + 0.5*dt, state + 0.5*dt*k1)
            k3 = rhs_tf(t + 0.5*dt, state + 0.5*dt*k2)
            k4 = rhs_tf(t + dt, state + dt*k3)
            state = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            t += dt
            if k % max(1, nsteps//50) == 0:
                traj.append(state.numpy())
        return np.array(traj)

    # end TF section

print('\nTranslation complete. If you want 2D/3D translations we can add them next.')

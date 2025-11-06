6.11.2025

ChatGPT: Done — I converted the Mathematica ( https://github.com/blue-science/subquantum_kinetics/blob/master/particle.nb ) notebook into a runnable Python script and
placed it in a code pane titled Particle Model G (you can see it next to the chat).

What I created

model_g_particle_nb_1a.py — a single-file translation that:

Implements the 1-D Model G PDE system (method of lines + scipy.integrate.solve_ivp) matching the Mathematica setup (L=100, T=30, eqs17 parameters).

Uses second-order finite differences for Laplacian/gradient and Dirichlet BCs (as in the notebook).

Reproduces the seed forcing chi(x,t) and steady-state values G0, X0, Y0.

Produces plots, saves frames and writes an MP4 animation (particle_model_g.mp4).

Includes an optional TensorFlow explicit RK4 integrator (if tensorflow is installed) that can run on GPU (explicit integrator requires small dt for stability).

How to run

Make sure you have the usual scientific Python stack:

pip install numpy scipy matplotlib imageio


To enable the TensorFlow option:

pip install tensorflow


Run:

python model_g_particle_nb_1a.py


The script will solve the PDE, save a final snapshot (out_frames/final.png), render PNG frames into out_frames/ and write particle_model_g.mp4.

Notes & limitations

I implemented the Mathematica notebook's 1-D case (the notebook has 2D/3D machinery, but the example run uses dimension==1). If you want 2D (cylindrical) or 3D (spherical) simulations I can extend the code (requires switching to radial Laplacians and different BCs — and becomes more computationally heavy).

The TensorFlow integrator is explicit RK4 (for portability to GPU). For stiff PDEs or large grids prefer implicit integrators (SciPy BDF or custom implicit schemes); I kept SciPy solve_ivp as the primary solver to match Mathematica's NDSolve.

Animation uses imageio to write MP4; ffmpeg will be used internally if available. Frames are stored in out_frames/ (you can delete them after the MP4 is created).

I preserved the notebook parameter values in params. You can tweak nx, max_step, or nt_anim for speed vs quality.

If you'd like:

I can extend this to 2D/3D (radial) as in the original notebook.

I can produce a Jupyter notebook version with interactive sliders for parameters/time.

I can inline a unit-test comparison or sample outputs to verify (e.g., reproduce specific plots from the paper).

Tell me which next step you want and I’ll update the code accordingly.

"""
Additional functions and demos for showcase notebook.
"""
import sys, os
from clawpack import pyclaw
from clawpack import riemann
from clawpack.pyclaw import examples
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from utils import riemann_tools


def euler_2d_quadrant_animation(numframes, resolution, sharp):
    """Plots animation of solution with bump initial condition, 
    using pyclaw (calls bump_pyclaw)."""
    xx, yy, frames = euler_2d_quadrant_pyclaw(numframes, resolution, sharp) 
    #fig = plt.figure(figsize=(4.5,3.0))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,3))
    # Set axes
    ax1.set_xlim(-0.75, 0.75)
    ax1.set_ylim(-0.5, 0.5)
    ax2.set_xlim(-0.75, 0.75)
    ax2.set_ylim(-0.5, 0.5)
    ax1.set_title('Density (schlieren)')
    ax2.set_title('Density')

    density = frames[0].q[0,:,:]
    (vx,vy) = np.gradient(density)
    vs = np.sqrt(vx**2 + vy**2)
    schlieren_plot = ax1.imshow(vs.T, cmap='Greys', vmin=vs.min(), vmax=vs.max()/20, 
           extent=[xx.min(), xx.max(), yy.min(), yy.max()],
           interpolation='nearest', origin='lower')
    density_plot = ax2.imshow(density, cmap='YlGnBu', vmin=density.min(), vmax=density.max(), 
           extent=[xx.min(), xx.max(), yy.min(), yy.max()],
           interpolation='nearest', origin='lower')
    fig.tight_layout()


    def fplot(frame_number):
        frame = frames[frame_number]
        density = frame.q[0,:,:]
        (vx,vy) = np.gradient(density)
        vs = np.sqrt(vx**2 + vy**2)

        schlieren_plot.set_data(vs.T)
        density_plot.set_data(density)
        return schlieren_plot,

    return animation.FuncAnimation(fig, fplot, frames=len(frames), interval=30)

def euler_2d_quadrant_pyclaw(numframes, resolution, sharp):
    """Returns pyclaw solution of bump initial condition."""
    # Set pyclaw for burgers equation 1D
    claw = pyclaw.Controller()
    claw.tfinal = 1.0           # Set final time
    claw.num_output_times = numframes  # Number of output frames
    claw.keep_copy = True       # Keep solution data in memory for plotting
    claw.output_format = None   # Don't write solution data to file
    if sharp:
        claw.solver = pyclaw.SharpClawSolver2D( riemann.euler_4wave_2D)
    else:
        claw.solver = pyclaw.ClawSolver2D( riemann.euler_4wave_2D)  # Choose Eulers 2D Riemann solver
    claw.solver.all_bcs = pyclaw.BC.extrap                  # Choose extrapolating BCs
    claw.verbosity = False                                 # Don't print pyclaw output
    grid_size = (int(3*resolution/2), resolution)
    domain = pyclaw.Domain( (-0.75, -0.5), (0.75, 0.5), grid_size)   # Choose domain and mesh resolution
    claw.solution = pyclaw.Solution(claw.solver.num_eqn,domain)
    ## Set initial condition
    gam = 1.4
    claw.solution.problem_data['gamma']  = gam             # Set gamma parameter
    xx, yy =domain.grid.p_centers
    q = claw.solution.q
    l = xx<0.0; r = xx>=0.0; b = yy<0.0; t = yy>=0.0
    q[0,...] = 2.*l*t + 1.*l*b + 1.*r*t + 3.*r*b
    q[1,...] = 0.75*t - 0.75*b
    q[2,...] = 0.5*l  - 0.5*r
    q[3,...] = 0.5*q[0,...]*(q[1,...]**2+q[2,...]**2) + 1./(gam-1.)     
    claw.solver.dt_initial = 1.e99

    # Run pyclaw
    status = claw.run()
    
    return xx, yy, claw.frames

def triplestate_animation(ql, qm, qr, numframes):
    """Plots animation of solution with triple-state initial condition, using pyclaw (calls  
    triplestate_pyclaw). Also plots characteristic structure by plotting contour plots of the 
    solution in the x-t plane """
    # Get solution for animation and set plot
    x, frames = triplestate_pyclaw(ql, qm, qr, numframes) 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,4))
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 5)
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(0, 2)
    ax1.set_title('Solution q(x)')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$q$')
    ax2.set_title('xt-plane')
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$t$')
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
    line1, = ax1.plot([], [], '-k', lw=2)

    # Contour plot of high-res solution to show characteristic structure in xt-plane
    meshpts = 2400
    numframes2 = 600
    x2, frames2 = triplestate_pyclaw(ql, qm, qr, numframes2) 
    characs = np.zeros([numframes2,meshpts])
    xx = np.linspace(-12,12,meshpts)
    tt = np.linspace(0,2,numframes2)
    for j in range(numframes2):
        characs[j] = frames2[j].q[0]
    X,T = np.meshgrid(xx,tt)
    ax2.contour(X, T, characs, levels=np.linspace(ql, ql+0.11 ,20), linewidths=0.5, colors='k')
    ax2.contour(X, T, characs, levels=np.linspace(qm+0.11, qm+0.13 ,7), linewidths=0.5, colors='k')
    ax2.contour(X, T, characs, levels=np.linspace(qr+0.13, qr+0.2 ,15), linewidths=0.5, colors='k')
    ax2.contour(X, T, characs, 12,  linewidths=0.5, colors='k')
    #ax2.contour(X, T, characs, 38, colors='k')
    # Add animated time line to xt-plane
    line2, = ax2.plot(x, 0*x , '--k')

    line = [line1, line2]

    # Update data function for animation
    def fplot(frame_number):
        frame = frames[frame_number]
        pressure = frame.q[0,:]
        line[0].set_data(x,pressure)
        line[1].set_data(x,0*x+frame.t)
        return line

    return animation.FuncAnimation(fig, fplot, frames=len(frames), interval=30, blit=False)

def triplestate_pyclaw(ql, qm, qr, numframes):
    """Returns pyclaw solution of triple-state initial condition."""
    # Set pyclaw for burgers equation 1D
    meshpts = 2400 #600
    claw = pyclaw.Controller()
    claw.tfinal = 2.0           # Set final time
    claw.keep_copy = True       # Keep solution data in memory for plotting
    claw.output_format = None   # Don't write solution data to file
    claw.num_output_times = numframes  # Number of output frames
    claw.solver = pyclaw.ClawSolver1D(riemann.burgers_1D)  # Choose burgers 1D Riemann solver
    claw.solver.all_bcs = pyclaw.BC.extrap               # Choose periodic BCs
    claw.verbosity = False                                # Don't print pyclaw output
    domain = pyclaw.Domain( (-12.,), (12.,), (meshpts,))   # Choose domain and mesh resolution
    claw.solution = pyclaw.Solution(claw.solver.num_eqn,domain)
    # Set initial condition
    x=domain.grid.x.centers
    q0 = 0.0*x
    xtick1 = 900 + int(meshpts/12)
    xtick2 = xtick1 + int(meshpts/12)
    for i in range(xtick1):
	    q0[i] = ql + i*0.0001
    #q0[0:xtick1] = ql
    for i in np.arange(xtick1, xtick2):
        q0[i] = qm + i*0.0001
    #q0[xtick1:xtick2] = qm
    for i in np.arange(xtick2, meshpts):
        q0[i] = qr + i*0.0001
    #q0[xtick2:meshpts] = qr
    claw.solution.q[0,:] = q0    
    claw.solver.dt_initial = 1.e99
    # Run pyclaw
    status = claw.run()
    
    return x, claw.frames





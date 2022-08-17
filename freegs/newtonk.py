"""
Routines for solving the nonlinear part of the Grad-Shafranov equation

This file has been added on August 11 2022
by Nicola C. Amorisco, George Holt, Adriano Agnello
to the original FreeGS code.

It provides an implementation of the Newton-Krylov algorithm to solve the pure forward GS problem, 
ie all coil currents are assigned and fixed, a solution for plasma_psi is sought

Differently from Picard iterations, use of the 'constrain' object is not supported
"""

import numpy as np
from numpy import amin, amax

# solves the least square problem min||Gmatrix.coeffs+residual||^2 with respect to coeffs: 
def best_coeff(residual, Gmatrix, clip=10):
    coeffs = np.matmul(np.matmul(np.linalg.inv(np.matmul(Gmatrix.T, Gmatrix)),
                                    Gmatrix.T), -residual.reshape(-1))
    return np.clip(coeffs, -clip, clip)

# refactoring the root problem F(plasma_psi) = tot_psi-eq.solve(tot_psi)
def F(eq, profiles, plasma_psi, tokamak_psi):
    # Solve equilbrium, using the given psi to calculate Jtor
    # this replaces eq.plasma_psi
    eq.solve(profiles, psi=plasma_psi+tokamak_psi)
    # tokamak psi is psi from all Machine coils, i.e.
    # tokamak_psi = eq.tokamak.calcPsiFromGreens(pgreen=eq._pgreen)

    # get residual and compare with previous trial_plasma_psi
    psi_change = eq.plasma_psi - plasma_psi
    return psi_change

def Arnoldi_iteration(eq,
                      profiles,
                      plasma_psi, 
                      dpsi_direction,
                      psi_change = None, 
                      tokamak_psi=None,
                      n_k=10, 
                      new_t=.5, 
                      grad_eps=.5,
                      clip=10,
                      verbose=False
                      ):
    """
    Routine to expand the Jacobian of F(plasma_psi) = tot_psi-eq.solve(tot_psi) 

    eq                - an Equilibrium object (equilibrium.py)
    profiles          - A Profile object for toroidal current (jtor.py)

    plasma_psi        - initial trial value of plasma_psi, ie the point of expansion
    dpsi_direction    - direction of the first term in the expansion: 
                        F(plasma_psi+alpha*dpsi_direction) for some appropriate alpha
                        It is customary to use dpsi_direction = psi_change = F(plasma_psi)
    psi_change        - value of F(plasma_psi), ie initial residual
    tokamak_psi       - psi from all Machine coils, i.e.
                        tokamak_psi = eq.tokamak.calcPsiFromGreens(pgreen=eq._pgreen)
    All psi-inputs above have shape = (nx,ny) = shape(eq.psi())

    n_k      - maximum number of terms in Jacobian expansion
    new_t    - add new term in Jacobian expansion if (relative) magnitude of the residual yet to explain is > new_t
    grad_eps - coefficient that sizes dpsi when calculating gradient terms
    clip     - clips expansion coefficients of the residual if larger in modulus than clip
    """

    nplasma_psi = np.linalg.norm(plasma_psi)
    nx,ny = np.shape(plasma_psi)

    #basis in Psi space
    Q = np.zeros((nx*ny, n_k+1))
    #orthonormal basis in Psi space
    Qn = np.zeros((nx*ny, n_k+1))
    #basis in F(psi) space
    G = np.zeros((nx*ny, n_k+1))
    
    # set tokamak_psi
    if tokamak_psi is None:
        tokamak_psi = eq.tokamak.calcPsiFromGreens(pgreen=eq._pgreen)
    
    # calculate first residual
    if psi_change is None:
        psi_change = F(eq, profiles, plasma_psi, tokamak_psi)
    lpsi_change = psi_change.reshape(-1)
    npsi_change = np.linalg.norm(lpsi_change)

    
    n_it = 0
    #control on whether to add a new basis vector
    arnoldi_control = 1
    #use at least 1 term, not more than n_k
    while arnoldi_control*(n_it<n_k)>0:
        # rescale dpsi to an appropriate magnitude
        alpha = grad_eps*nplasma_psi/np.linalg.norm(dpsi_direction)*npsi_change
        candidate_dpsi = alpha*dpsi_direction
        ri = F(eq, profiles, plasma_psi + candidate_dpsi, tokamak_psi)
        candidate_usable = ri - psi_change
        lvec_direction = candidate_usable.reshape(-1)

        # store used dpsi
        Q[:,n_it] = candidate_dpsi.reshape(-1)
        # store normalized dpsi
        ncandidate_dpsi = np.linalg.norm(Q[:,n_it])
        Qn[:,n_it] = Q[:,n_it]/ncandidate_dpsi
        
        # store residual term 
        G[:,n_it] = lvec_direction
        n_it += 1
        #orthogonalize new dpsi_direction for next loop  
        lvec_direction -= np.sum(np.sum(Qn[:,:n_it]*lvec_direction[:,np.newaxis], axis=0, keepdims=True)*Qn[:,:n_it], axis=1)
        # prepare dpsi_direction for next loop
        dpsi_direction = lvec_direction.reshape(nx,ny)

        #check if more terms are needed
        #arnoldi_control = (np.linalg.norm(vec_direction)/nFresidual > conv_crit)
        coeffs = best_coeff(psi_change, G[:,:n_it], clip=clip)
        explained_residual = np.sum(G[:,:n_it]*coeffs[np.newaxis,:], axis=1)    
        relative_unexpl_residual = np.linalg.norm(explained_residual+lpsi_change)/npsi_change
        arnoldi_control = (relative_unexpl_residual > new_t)
        if verbose-2>0:
            print('this is Arnoldi term', n_it)
            print('norm(dpsi) = ', ncandidate_dpsi)
            print('norm(usable_residual)/norm(target_residual) = ', np.linalg.norm(candidate_usable)/npsi_change)
            print('relative_unexplained_residual = ', relative_unexpl_residual)

    if verbose-1>0:
        print('Arnoldi coefficients:', coeffs)
    best_dpsi = np.sum(Q[:,:n_it]*coeffs[np.newaxis,:], axis=1).reshape(nx,ny)
    return best_dpsi


def NKsolve(eq, 
          profiles,
          rtol=1e-6,
          atol=1e-10,
          show=False,
          axis=None,
          pause=0.0001,
          n_k=8,
          new_t=.5, 
          grad_eps=.5,
          clip=10,
          maxits=30,
          verbose=False,
          convergenceInfo=False
          ):

    """
    eq       - an Equilibrium object (equilibrium.py)
    profiles - A Profile object for toroidal current (jtor.py)

    rtol     - Relative tolerance (change in psi)/( max(psi) - min(psi) )

    show     - If true, plot the plasma equilibrium at each nonlinear step
    axis     - Specify a figure to plot onto. Default (None) creates a new figure
    pause    - Delay between output plots. If negative, waits for window to be closed

    n_k      - maximum number of terms in Jacobian expansion
    new_t    - add new term in Jacobian expansion if (relative) magnitude of the residual yet to explain is > new_t
    grad_eps - coefficient that sizes dpsi when calculating gradient terms
    clip     - clips expansion coefficients of the residual if larger in modulus than clip

    maxits   - Maximum number of iterations. Set to None for no limit.
               If this limit is exceeded then a RuntimeError is raised.
    
    verbose  - if True it outputs progress on the realtive convergence
               if >1 it outputs used coefficients in Arnoldi expansion
               if >2 it outputs detailed info on the Jacobian expansion
    """
    if show:
        import matplotlib.pyplot as plt
        from .plotting import plotEquilibrium

        if pause > 0.0 and axis is None:
            # No axis specified, so create a new figure
            fig = plt.figure()
            axis = fig.add_subplot(111)
    
    # save psi from active coils
    tokamak_psi = eq.tokamak.calcPsiFromGreens(pgreen=eq._pgreen)
    # get initial plasma_psi
    trial_plasma_psi = eq.plasma_psi.copy()

    iteration = 0  # Count number of iterations
    psi_maxchange_iterations, psi_relchange_iterations = [], []
    # Start main loop
    while True:
        if show:
            # Plot state of plasma equilibrium
            if pause < 0:
                fig = plt.figure()
                axis = fig.add_subplot(111)
            else:
                axis.clear()

            plotEquilibrium(eq, axis=axis, show=False)

            if pause < 0:
                # Wait for user to close the window
                plt.show()
            else:
                # Update the canvas and pause
                # Note, a short pause is needed to force drawing update
                axis.figure.canvas.draw()
                plt.pause(pause)
        
        
        # get residual and compare with previous trial_plasma_psi
        psi_change = F(eq, profiles, trial_plasma_psi, tokamak_psi)
        psi_maxchange = amax(abs(psi_change))
        psi_relchange = psi_maxchange / (amax(trial_plasma_psi) - amin(trial_plasma_psi))
        if verbose:
            print('this is iteration', iteration)
            print('psi_relchange = ', psi_relchange)
            
        # Check if the relative change in psi is small enough
        if (psi_maxchange < atol) or (psi_relchange < rtol):
            break
        
        # store progress
        psi_maxchange_iterations.append(psi_maxchange)
        psi_relchange_iterations.append(psi_relchange)
        
        
        # if relative change is too large, use simple Picard, 
        # only start with actual NK when psi_relchange<.05
        if psi_relchange>.05:
            #this is Picard iteration
            trial_plasma_psi += psi_change
            if verbose:
                print('Using Picard here!')
        else:
            best_dpsi = Arnoldi_iteration(eq, profiles, 
                                        trial_plasma_psi, 
                                        dpsi_direction=psi_change,
                                        psi_change=psi_change,
                                        tokamak_psi=tokamak_psi,
                                        n_k=n_k,
                                        new_t=new_t,
                                        grad_eps=grad_eps,
                                        clip=clip,
                                        verbose=verbose)
            trial_plasma_psi += best_dpsi

        # Check if the maximum iterations has been exceeded
        iteration += 1
        if maxits and iteration > maxits:
            raise RuntimeError(
                "NewtonK iteration failed to converge (too many iterations)"
            )    

    # make critical points accessible
    eq.opoint = profiles.opt
    eq.xpoint = profiles.xpt

    if convergenceInfo: 
        return np.array(psi_maxchange_iterations),\
               np.array(psi_relchange_iterations)
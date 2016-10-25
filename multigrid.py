"""
Multigrid solver for elliptic problems

Example
-------

$ python multigrid.py

This will run the 

"""

from numpy import zeros,max,amin,amax, abs

def smoothJacobi(A, x, b, dx, dy):
    """
    Smooth the solution using Jacobi method
    """
    
    if b.shape != x.shape:
        raise ValueError("b and x have different shapes")
    
    smooth = x + ( b -  A(x, dx, dy) ) / A.diag(dx, dy)
    
    return smooth
    
 
def restrict(orig, dx, dy, coarse=None, avg=False):
    """
    Coarsen the original onto a coarser mesh

    Inputs
    ------
    
    orig[nx,ny] - A 2D numpy array. Each dimension must have
                  a size (2^n + 1) though nx != ny is possible

    Returns
    -------

    A 2D numpy array of size [(nx-1)/2+1, (ny-1)/2+1]
    """
    
    nx = orig.shape[0]
    ny = orig.shape[1]
    
    if (nx-1) % 2 == 1 or (ny-1) % 2 == 1:
        # Can't divide any further
        if coarse == None:
            return orig
        coarse.resize(orig.shape)
        coarse[:,:] = orig
        return
    
    # Dividing x and y in 2
    nx = (nx-1) / 2  + 1
    ny = (ny-1) / 2  + 1
    
    if coarse == None:
        coarse = zeros([nx,ny])
    else:
        coarse.resize([nx,ny])
        
    for x in range(1,nx-1):
        for y in range(1,ny-1):
            x0 = 2*x
            y0 = 2*y
            coarse[x,y] = orig[x0,y0]/4.
            + (orig[x0+1,y0] + orig[x0-1,y0] + orig[x0,y0+1] + orig[x0,y0-1])/8.
            + (orig[x0-1,y0-1] + orig[x0-1,y0+1] + orig[x0+1,y0-1] + orig[x0+1,y0+1])/16.
    if not avg:
        coarse *= 4.
        
    return coarse, dx*2., dy*2.

def interpolate(orig):
    """
    Interpolate a solution onto a finer mesh
    """
    nx = orig.shape[0]
    ny = orig.shape[1]
    
    nx2 = 2*(nx-1) + 1
    ny2 = 2*(ny-1) + 1
    
    fine = zeros([nx2,ny2])
    
    for x in range(1,nx-1):
        for y in range(1,ny-1):
            x0 = 2*x
            y0 = 2*y
            
            fine[x0-1,y0-1] += 0.25*orig[x,y]
            fine[x0-1,y0  ] += 0.5*orig[x,y]
            fine[x0-1,y0+1] += 0.25*orig[x,y]
            
            fine[x0  ,y0-1] += 0.5*orig[x,y]
            fine[x0  ,y0  ] = orig[x,y]
            fine[x0  ,y0+1] += 0.5*orig[x,y]
            
            fine[x0+1,y0-1] += 0.25*orig[x,y]
            fine[x0+1,y0  ] += 0.5*orig[x,y]
            fine[x0+1,y0+1] += 0.25*orig[x,y]
            
    return fine
            
    
def smoothVcycle(A, x, b, dx, dy, niter=10, sublevels=0):
    """
    Perform smoothing using multigrid
    """
    
    # Smooth
    for i in range(niter):
        x = smoothJacobi(A, x, b, dx, dy)
    
    if sublevels > 0:
        # Calculate the error
        error = b - A(x, dx, dy)
        
        # Restrict error onto coarser mesh
        Cerror, Cdx, Cdy = restrict(error, dx, dy)
        
        # smooth this error
        Cx = zeros(Cerror.shape)
        Cx = smoothVcycle(A, Cx, Cerror, Cdx, Cdy, niter, sublevels-1)
        
        # Prolong the solution
        xupdate = interpolate(Cx)
        
        x = x + xupdate
        
        errup = A(xupdate, dx, dy)

        #plt.figure()
        #plt.plot(error[2,:],label="error")
        #plt.plot(Cerror[2,:],label="Cerror")
        #plt.plot(errup[2,:],label="errup")
        #plt.legend()
        #plt.show()
        
    # Smooth
    for i in range(niter):
        x = smoothJacobi(A, x, b, dx, dy)
    
    return x
    
def smoothMG(A, x, b, dx, dy, niter=10, sublevels=1, ncycle=2):
    error = b - A(x, dx, dy)
    print "Starting max residual: ", max(abs(error))
    
    for c in range(ncycle):
        x = smoothVcycle(A, x, b, dx, dy, niter, sublevels)
        
        error = b - A(x, dx, dy)
        print "Cycle ", c, ": " , max(abs(error))
    return x

########################################

class LaplacianOp:
    """
    Implements a simple Laplacian operator
    for use with the multigrid solver
    """
    def __call__(self, f, dx, dy):
        nx = f.shape[0]
        ny = f.shape[1]

        b = zeros([nx,ny])

        for x in range(1,nx-1):
            for y in range(1,ny-1):
                # Loop over points in the domain

                b[x,y] = (f[x-1,y] - 2*f[x,y] + f[x+1,y])/dx**2 + (f[x,y-1] - 2*f[x,y] + f[x,y+1])/dy**2

        return b

    def diag(self, dx, dy):
        return -2./dx**2 - 2./dy**2

if __name__ == "__main__":

    # Test case

    from numpy import meshgrid, exp, linspace
    import matplotlib.pyplot as plt

    nx = 65
    ny = 65

    dx = 1./nx
    dy = 1./ny

    xx, yy = meshgrid(linspace(0,1,nx), linspace(0,1,ny))

    rhs = exp( - ( (xx - 0.5)**2 + (yy - 0.5)**2 ) / 0.4**2 )

    rhs[0,:] = 0.0
    rhs[:,0] = 0.0
    rhs[nx-1,:] = 0.0
    rhs[:,ny-1] = 0.0

    x = zeros([nx,ny])

    x2 = x.copy()

    A = LaplacianOp()

    ################ SIMPLE ITERATIVE SOLVER ##############

    for i in range(100):
      x2 = smoothJacobi(A, x, rhs, dx,dy)      
      x,x2 = x2, x # Swap arrays
      
      error = rhs - A(x, dx, dy)
      print i, max(abs(error))

    ################ MULTIGRID SOLVER #######################

    x = zeros([nx,ny])
    x = smoothMG(A, x, rhs, dx, dy, niter=5, sublevels=4, ncycle=2)
    
    f = plt.figure()
    plt.contourf(x)
    plt.show()

"""
Tests of the linear solver
"""

import numpy as np


###### Laplacian in 2D


def run():
    nx = 65
    ny = 65

    dx = 1./(nx - 1)
    dy = 1./(ny - 1)
    
    xx, yy = meshgrid(np.linspace(0,1,nx), np.linspace(0,1,ny))
    
    solution = np.exp( - ( (xx - 0.5)**2 + (yy - 0.5)**2 ) / 0.4**2 )

import numpy as np
from numba import jit

@jit(nopython=True)
def h_calc(h, Node, i, j):
    if Node[i,j] == 0:
        # internal point
        h_new = (h[i+1,j] + h[i-1,j] + h[i, j+1] + h[i, j-1])/4
    elif Node[i,j] == 1:
        # fixed head BC
        h_new = h[i, j]
    elif Node[i,j] == 21:
        # no flux left
        h_new = (2*h[i+1,j] + h[i, j+1] + h[i, j-1])/4
    elif Node[i, j] == 22:
        # no flux right
        h_new = (2*h[i-1,j] + h[i, j+1] + h[i, j-1])/4
    elif Node[i,j] == 23:
        # no flux bottom
        h_new = (h[i+1,j] + h[i-1,j] + 2*h[i, j+1])/4
    elif Node[i,j] == 24:
        # no flux top
        h_new = (h[i+1,j] + h[i-1,j] + 2*h[i, j-1])/4
    elif Node[i,j] == 31:
        # no flux lower left
        h_new = (2*h[i+1,j] + 2*h[i, j+1] )/4
    elif Node[i,j] == 32:
        # upper left
        h_new = (2*h[i+1,j] + 2*h[i, j-1])/4
    elif Node[i,j] == 33:
        # no flux lower right
        h_new = (2*h[i-1,j] + 2*h[i, j+1])/4
    elif Node[i,j] == 34:
        # no flux upper right
        h_new = (2*h[i-1,j] + 2*h[i, j-1])/4
    return h_new
    

@jit(nopython=True)
def GS_node(h, Node, tol):
    """BCs expressed by Node"""
    Nx, Ny = np.shape(h)
    lamb = 1.5
    max_dif = tol+1
    iters = 0
    while max_dif > tol and iters < 10000:
        iters += 1
        # loop over i and j
        max_dif = 0
        for j in range(0,Ny):
            for i in range(0,Nx):
                h_old = h[i,j]
                h_new = h_calc(h, Node, i, j)
                h_new = lamb*h_new + (1-lamb)*h_old
                h[i,j] = h_new
                h_dif = abs(h_new - h_old)
                max_dif = max(max_dif, h_dif)
    if iters < 10000:
        return h
    else:
        h[:,:] = np.nan
        return h

if __name__ == '__main__':
    Nx, Ny = int(200/2+1), int(110/2+1)
    h_init = np.ones( (Nx, Ny))*100
    h_init[0,1:], h_init[1:,0] = 150, 50 # left and bottom fixed head BCs
    Node = np.zeros( (Nx, Ny), dtype='int')
    Node[0,:], Node[:,0] = 1, 1 # left and bottom fixed head BCs
    Node[1:-1,-1] = 24 # no flux top
    Node[-1,-1] = 34 # no flux upper right
    Node[-1,1:-1] = 22 # no flux right
    h_arr = GS_node(h_init, Node, tol=1)
    print(h_arr.T)
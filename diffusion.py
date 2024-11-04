#1/usr/bin/env python3
'''
Tools and methods for solving our heat equation/diffusion
'''

import numpy as np
import matplotlib.pyplot as plt

# Solution to problem 10.3 from fink/matthews
sol10p3 = [[0.000000, 0.640000, 0.960000, 0.960000, 0.640000, 0.000000],
           [0.000000, 0.480000, 0.800000, 0.800000, 0.480000, 0.000000],
           [0.000000, 0.400000, 0.640000, 0.640000, 0.400000, 0.000000],
           [0.000000, 0.320000, 0.520000, 0.520000, 0.320000, 0.000000],
           [0.000000, 0.260000, 0.420000, 0.420000, 0.260000, 0.000000],
           [0.000000, 0.210000, 0.340000, 0.340000, 0.210000, 0.000000],
           [0.000000, 0.170000, 0.275000, 0.275000, 0.170000, 0.000000],
           [0.000000, 0.137500, 0.222500, 0.222500, 0.137500, 0.000000],
           [0.000000, 0.111250, 0.180000, 0.180000, 0.111250, 0.000000],
           [0.000000, 0.090000, 0.145625, 0.145625, 0.090000, 0.000000],
           [0.000000, 0.072812, 0.117813, 0.117813, 0.072812, 0.000000]]
sol10p3 = np.array(sol10p3).transpose()


def heatdiff(xmax=1, tmax=.2, dx=.2, dt=.02, c2=1, neumann=False, debug=False):
    '''
    Parameters:
    -----------
    xmax : float, defaults to 1
        Set the domain upper boundary location. In meters.
    tmax : float, defaults to .2s
        Set the domain time limit in seconds.
    dx: int
        The change in ground depth in m.
    dt: int
        The change in time step in seconds by default, measured in days
        when the model is applied to Greenland.
    c2: int, default=1
        diffusivity constant, in m^2/s by default, 
        in m2/day when the model is applied to Greenland. 
    debug: boolean, default=False
        Turn on debug output. 
    neumann : bool, defaults to False
        Switch to Neumann boundary conditions if true where dU/dx = 0
        Default behavior is Dirichlet where U=0 at boundaries.

    Returns:
    --------
    xgrid: Numpy array of size M
        Array of ground depths from 0m (at the surface) to xmax.
    tgrid: Numpy array of size N
        Array of times from 0 to tmax.
    U: Numpy array of temperatures in °C.
        Array of temperatures from the surface of the earth to at xmax.
    '''

    # Start by calculating size of array: MxN
    M = int(np.round(xmax / dx + 1))
    N = int(np.round(tmax / dt + 1))

    xgrid, tgrid = np.arange(0, xmax+dx, dx), np.arange(0, tmax+dt, dt)

    #Debugging commands utilized when debug is set to true:
    if debug:
        print(f'Our grid goes from 0 to {xmax}m and 0 to {tmax}s')
        print(f'Our spatial step is {dx} and time step is {dt}')
        print(f'There are {M} points in space and {N} points in time.')
        print('Here is our spatial grid:')
        print(xgrid)
        print('Here is our time grid:')
        print(tgrid)

    # Initialize our data array:
    U = np.zeros([M, N])

    # Set initial conditions:
    U[:, 0] = 4*xgrid - 4*xgrid**2

    # Set boundary conditions:
    U[0, :] = 0
    U[-1, :] = 0

    # Set our "r" constant.
    r = c2 * dt / dx**2

    # Solve! Forward differnce ahoy.
    for j in range(N-1):
        U[1:-1, j+1] = (1-2*r) * U[1:-1, j] + \
            r*(U[2:, j] + U[:-2, j])
        # Set Neumann-type boundary conditions:
        if neumann:
            U[0, j+1] = U[1, j+1]
            U[-1, j+1] = U[-2, j+1]

    # Return grid and result:
    return xgrid, tgrid, U

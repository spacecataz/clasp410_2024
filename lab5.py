#!/usr/bin/env python3

'''
Draft code for Lab 5: SNOWBALL EARTH!!!
'''

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# Some constants:
radearth = 6357000.  # Earth radius in meters.
mxdlyr = 50.         # depth of mixed layer (m)
sigma = 5.67e-8      # Steffan-Boltzman constant
C = 4.2e6            # Heat capacity of water
rho = 1020           # Density of sea-water (kg/m^3)


def gen_grid(nbins=18):
    '''
    Generate a grid from 0 to 180 lat (where 0 is south pole, 180 is north)
    where each returned point represents the cell center.

    Parameters
    ----------
    nbins : int, defaults to 18
        Set the number of latitude bins.

    Returns
    -------
    dlat : float
        Grid spacing in degrees.
    lats : Numpy array
        Array of cell center latitudes.
    '''

    dlat = 180 / nbins  # Latitude spacing.
    lats = np.arange(0, 180, dlat) + dlat/2.

    # Alternative way to obtain grid:
    # lats = np.linspace(dlat/2., 180-dlat/2, nbins)

    return dlat, lats


def temp_warm(lats_in):
    '''
    Create a temperature profile for modern day "warm" earth.

    Parameters
    ----------
    lats_in : Numpy array
        Array of latitudes in degrees where temperature is required

    Returns
    -------
    temp : Numpy array
        Temperature in Celcius.
    '''

    # Get base grid:
    dlat, lats = gen_grid()

    # Set initial temperature curve:
    T_warm = np.array([-47, -19, -11, 1, 9, 14, 19, 23, 25, 25,
                       23, 19, 14, 9, 1, -11, -19, -47])
    coeffs = np.polyfit(lats, T_warm, 2)

    # Now, return fitting:
    temp = coeffs[2] + coeffs[1]*lats_in + coeffs[0] * lats_in**2

    return temp


def snowball_earth(nbins=18, dt=1., tstop=10000, lam=100., debug=False):
    '''
    Perform snowball earth simulation.

    Parameters
    ----------
    nbins : int, defaults to 18
        Number of latitude bins.
    dt : float, defaults to 1
        Timestep in units of years
    tstop : float, defaults to 10,000
        Stop time in years
    lam : float, defaults to 100
        Diffusion coefficient of ocean in m^2/s
    debug : bool, defaults to False
        Turn  on or off debug print statements.

    Returns
    -------
    lats : Numpy array
        Latitude grid in degrees where 0 is the south pole.
    Temp : Numpy array
        Final temperature as a function of latitude.
    '''
    # Get time step in seconds:
    dt_sec = 365 * 24 * 3600 * dt  # Years to seconds.

    # Generate grid:
    dlat, lats = gen_grid(nbins)

    # Get grid spacing in meters.
    dy = radearth * np.pi * dlat / 180.

    # Create initial condition:
    Temp = temp_warm(lats)
    if debug:
        print('Initial temp = ', Temp)

    # Get number of timesteps:
    nstep = int(tstop / dt)

    # Debug for problem initialization
    if debug:
        print("DEBUG MODE!")
        print(f"Function called for nbins={nbins}, dt={dt}, tstop={tstop}")
        print(f"This results in nstep={nstep} time step")
        print(f"dlat={dlat} (deg); dy = {dy} (m)")
        print("Resulting Lat Grid:")
        print(lats)

    # Build A matrix:
    if debug:
        print('Building A matrix...')
    A = np.identity(nbins) * -2  # Set diagonal elements to -2
    A[np.arange(nbins-1), np.arange(nbins-1)+1] = 1  # Set off-diag elements
    A[np.arange(nbins-1)+1, np.arange(nbins-1)] = 1  # Set off-diag elements
    # Set boundary conditions:
    A[0, 1], A[-1, -2] = 2, 2

    if debug:
        print('A = ', A)
    # Set units of A derp
    A /= dy**2

    # Get our "L" matrix:
    L = np.identity(nbins) - dt_sec * lam * A
    L_inv = np.linalg.inv(L)

    if debug:
        print('Time integrating...')
    for i in range(nstep):
        Temp = np.matmul(L_inv, Temp)

    return lats, Temp


def test_snowball(tstop=10000):
    '''
    Reproduce example plot in lecture/handout.

    Using our DEFAULT values (grid size, diffusion, etc.) and a warm-Earth
    initial condition, plot:
        - Initial condition
        - Plot simple diffusion only
        - Plot simple diffusion + spherical correction
        - Plot simple diff + sphere corr + insolation
    '''
    nbins = 18

    # Generate very simple grid
    # Generate grid:
    dlat, lats = gen_grid(nbins)

    # Create initial condition:
    initial = temp_warm(lats)

    # Get simple diffusion solution:
    lats, t_diff = snowball_earth(tstop=tstop, debug=True)

    # Create figure and plot!
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(lats, initial, label='Warm Earth Init. Cond.')
    ax.plot(lats, t_diff, label='Simple Diffusion')

    ax.set_xlabel('Latitude (0=South Pole)')
    ax.set_ylabel('Temperature ($^{\circ} C$)')

    ax.legend(loc='best')

    fig.tight_layout()
#!/usr/bin/env python3

'''
This file performs fire/disease spread simulations.

To get solution for lab 1: Run these commands:

>>> blah
>>> blah blah

'''

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')


def fire_spread(nNorth=3, nEast=3, maxiter=4, pspread=1.0):
    '''
    This function performs a fire/disease spread simultion.

    Parameters
    ==========
    nNorth, nEast : integer, defaults to 3
        Set the north-south (i) and east-west (j) size of grid.
        Default is 3 squares in each direction.
    maxiter : int, defaults to 4
        Set the maximum number of iterations including initial condition
    pspread : float, defaults to 1
        Chance fire spreads from 0 to 1 (0 to 100%).
    '''

    # Create forest and set initial condition
    forest = np.zeros([maxiter, nNorth, nEast]) + 2

    # Set fire! To the center of the forest.
    istart, jstart = nNorth//2, nEast//2
    forest[0, istart, jstart] = 3

    # Plot initial condition
    fig, ax = plt.subplots(1, 1)
    contour = ax.matshow(forest[0, :, :], vmin=1, vmax=3)
    ax.set_title(f'Iteration = {0:03d}')
    plt.colorbar(contour, ax=ax)

    # Propagate the solution.
    for k in range(maxiter-1):
        # Set chance to burn:
        ignite = np.random.rand(nNorth, nEast)

        # Use current step to set next step:
        forest[k+1, :, :] = forest[k, :, :]

        # Burn from north to south:
        doburn = (forest[k, :-1, :] == 3) & (forest[k, 1:, :] == 2) & \
            (ignite[1:, :] <= pspread)
        forest[k+1, 1:, :][doburn] = 3

        # Burn in each cardinal direction.
        # From south to north:
        for i in range(1, nNorth):
            for j in range(nEast):
                # Is current patch burning AND adjacent forested?
                if (forest[k, i, j] == 3) & (forest[k, i-1, j] == 2):
                    # Spread fire to new square:
                    forest[k+1, i-1, j] = 3

        # From east to west
        for i in range(nNorth):
            for j in range(nEast-1):
                # Is current patch burning AND adjacent forested?
                if (forest[k, i, j] == 3) & (forest[k, i, j+1] == 2):
                    # Spread fire to new square:
                    forest[k+1, i, j+1] = 3

        # From west to east
        for i in range(nNorth):
            for j in range(1, nEast):
                # Is current patch burning AND adjacent forested?
                if (forest[k, i, j] == 3) & (forest[k, i, j-1] == 2):
                    # Spread fire to new square:
                    forest[k+1, i, j-1] = 3

        # Set currently burning to bare:
        wasburn = forest[k, :, :] == 3  # Find cells that WERE burning
        forest[k+1, wasburn] = 1       # ...they are NOW bare.

        fig, ax = plt.subplots(1, 1)
        contour = ax.matshow(forest[k+1, :, :], vmin=1, vmax=3)
        ax.set_title(f'Iteration = {k+1:03d}')
        plt.colorbar(contour, ax=ax)

        fig.savefig(f'fig{k:04d}.png')
        plt.close('all')

        # Quit if no spots are on fire.
        nBurn = (forest[k+1, :, :] == 3).sum()
        if nBurn == 0:
            print(f"Burn completed in {k+1} steps")
            break

    return k+1


def explore_burnrate():
    ''' Vary burn rate and see how fast fire ends.'''

    prob = np.arange(0, 1, .05)
    nsteps = np.zeros(prob.size)

    for i, p in enumerate(prob):
        print(f"Buring for pspread = {p}")
        nsteps[i] = fire_spread(nEast=5, pspread=p, maxiter=100)

    plt.plot(prob, nsteps)
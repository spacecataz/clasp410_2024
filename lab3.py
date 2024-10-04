#!/usr/bin/env python3
'''
A set of tools and routines for solving the N-layer atmosphere energy
balance problem and perform some useful analysis.

TO REPRODUCE FIGURES IN LAB WRITE-UP: DO THE FOLLOWING
blah
blah
blah
'''

import numpy as np
import matplotlib.pyplot as plt

 # Define some useful constants here.
sigma = 5.67E-8  # Steffan-Boltzman constant.


def n_layer_atmos(N, epsilon, S0=1350, albedo=0.33):
    '''
    Solve the n-layer atmosphere problem and return temperature at each layer.

    Parameters
    ----------
    N : int
        Set the number of layers.
    epsilon : float, default=1.0
        Set the emissivity of the atmospheric layers.
    albedo : float, default=0.33
        Set the planetary albedo from 0 to 1.
    S0 : float, default=1350
        Set the incoming solar shortwave flux in Watts/m^2.
    debug : boolean, default=False
        Turn on debug output.
    '''

    # Create matrices:
    A = np.zeros([N+1, N+1])
    b = np.zeros(N+1)
    b[0] = -S0/4 * (1-albedo)

    # Populate our A matrix.
    for i in range(N+1):
        for j in range(N+1):
            if i == j:
                A[i, j] = -1*(i > 0) - 1
            else:
                m = np.abs(j-i) - 1
                A[i, j] = epsilon * (1-epsilon)**m
    A[0, 1:] /= epsilon
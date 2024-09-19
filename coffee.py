#!/usr/bin/env python3

'''
This file produces the solution to the coffe problem which is really
important to everyone.

To use this file....

This file solves Newton's law of cooling for ...

The following table sets the values and their units used throughout this code:

| Symbol | Units  | Value/Meaning                                            |
|--------|--------|----------------------------------------------------------|
|T(t)    | C or K | Surface temperature of body in question                  |
|T_init  | C or K | Initial temperature of body in question                  |
|T_env   | C or K | Temperature of the ambient environment                   |
|k       | 1/s    | Heat transfer coefficient                                |
|t       | s      | Time in seconds
'''

import numpy as np
import matplotlib.pyplot as plt

# Set MPL style sheet
plt.style.use('fivethirtyeight')

# Define constants for our problem.
k = 1/300  # Coeff. of cooling in units of 1/s.
T_init, T_env = 90, 20  # initial and environmental temps in units of C.


def solve_temp(time, k=k, T_init=T_init, T_env=T_env):
    '''
    For a given scalar or array of times, `t`, return the analytic solution
    for Newton's law of cooling:

    $T(t)=T_env + \left( T(t=0) - T_{env} \right) e^{-kt}$

    ...where all values are defined in the docstring for this module.

    Parameters
    ==========
    time : Numpy array
        Array of times, in seconds, for which solution will be provided.

    Other Parameters
    ================
    k : float
        Heat transfer coefficient, defaults to 1/300. s^-1
    T_env : float
        Ambient environment temperature, defaults to 20°C.
    T_init : float
        Initial temperature of cooling object/mass, defaults to °C

    Returns
    =======
    temp : numpy array
        An array of temperatures corresponding to `t`.
    '''

    return T_env + (T_init - T_env) * np.exp(-k*time)


def time_to_temp(T_target, k=1/300., T_env=20, T_init=90):
    '''
    Given an initial temperature, `T_init`, an ambient temperature, `T_env`,
    and a cooling rate, return the time required to reach a target temperature,
    `T_target`.

        Parameters
    ==========
    T_target : scalar or numpy array
        Target temperature in °C.


    Other Parameters
    ================
    k : float
        Heat transfer coefficient, defaults to 1/300. s^-1
    T_env : float
        Ambient environment temperature, defaults to 20°C.
    T_init : float
        Initial temperature of cooling object/mass, defaults to °C

    Returns
    =======
    t : scalar or numpy array
        Time in s to reach the target temperature(s).
    '''

    return (-1/k) * np.log((T_target - T_env)/(T_init - T_env))


def newtcool(t, T0):
    '''Return dT/dt at time t using Newton's law of cooling'''
    return -k * (T0 - T_env)


def euler(diffq, f0, dt, tmax=600.):
    '''
    Given a function representing the first derivative of f, an
    initial condiition f0 and a timestep, dt, solve for f using
    Euler's method.
    '''

    # Initialize!
    t = np.arange(0.0, tmax+dt, dt)
    f = np.zeros(t.size)
    f[0] = f0

    # Integrate!
    for i in range(t.size-1):
        f[i+1] = f[i] + dt * diffq(t[i], f[i])

    # Return values to caller:
    return t, f


# Set up time array.
time = np.arange(0, 600, 1)

# Get temperature for first scenario: no cream until 60 deg.
temp_scn1 = solve_temp(time)     # Temp vs time
time_scn1 = time_to_temp(60.0)   # Time to reach 60C from T_init

# Repeat for scenario 2: add cream immediately.
temp_scn2 = solve_temp(time, T_init=85)     # Temp vs time
time_scn2 = time_to_temp(60.0, T_init=85)   # Time to reach 60C from T_init

# Get numerical approx:
time_euler, temp_euler = euler(newtcool, 90.0, dt=10)

# Plot! :)
fig, ax = plt.subplots(1, 1)

# Add temp vs. time
ax.plot(time, temp_scn1, label='Scenario 1')
ax.plot(time, temp_scn2, label='Scenario 2')
ax.plot(time_euler, temp_euler, label='Euler Approx')

# Add vert lines:
ax.axvline(time_scn1, ls='--', c='C0', label=r'$t(T=60^{\circ})$')
ax.axvline(time_scn2, ls='--', c='C1', label=r'$t(T=60^{\circ})$')

ax.legend(loc='best')
fig.savefig('coffee_solution.png')


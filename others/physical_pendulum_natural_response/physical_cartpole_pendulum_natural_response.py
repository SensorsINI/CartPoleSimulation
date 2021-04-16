"""
This script finds the parameters of cartpole pole
We assume that cartpole is immobilized. The equations boils down to:
angleDD * moment_of_inertia = -mg(L/2)sin(angle) - J_friction * angleD
moment_of_inertia = k*m*L^2
angleDD = (-g/(2kL)) * sin(angle) - J_friction/(kmL^2) * angleD

This we can write as
angleDD = a * sin(angle) + b * angleD

with small angle approx.:
angleDD = a * angle + b * angleD

# This gives (Mathematica):
DSolve[\[Theta]''[t] == a*\[Theta][t] + b*\[Theta]'[t], \[Theta][t],t]
\[Theta][t] ->
   E^(1/2 (b - Sqrt[4 a + b^2]) t) C[1] +
    E^(1/2 (b + Sqrt[4 a + b^2]) t) C[2]

Let us assume:
"""
g = 9.81
L = 0.395  # Full length of pole!
k = 1.0/3.0
m = 0.087
"""
We expect
a = -g/(2*k*L) = -37.25
b = - J_friction/(k*m*(L**2)) = -J_friction*221.01

For analysis which results are described in comment we used:
cartpole-2021-03-29-13-00-01-pole-natural-response.csv
Here cartpole was not fixed to the track, but we assume friction was big enough to assume so.

We found:
a = [-37.43,-38.11]
b = [-0.167, -0.054]
J_fric = [2.4e-4, 7.5e-4]

I suggest to take:
a = -37.43
b = -0.055
J_fric = 2.5e-4

Small angle approx.: This gives:
\[Theta][t] -> 
   E^(-0.0275 t) (C[2] Cos[6.11794 t] + C[1] Sin[6.11794 t])

T = 2*np.pi/6.11794 = 1.027 s

For comparison neglecting friction:
T = 2*np.pi/6.11801 = 1.027 s -> Friction has little impact on T

For comparison with maximal predicted a (-38.11)
T = 2*np.pi/6.17277 = 1.017
"""


# Change backend for matplotlib to plot interactively in pycharm
# This import must go before pyplot
from matplotlib import use
# use('TkAgg')
use('macOSX')

# Filter to smooth data
from scipy.signal import savgol_filter

# Find parameters of pendulum ODE - regression
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

file_path = './others/physical_pendulum_natural_response/cartpole-2021-03-29-13-00-01-pole-natural-response.csv'

# Load data
data: pd.DataFrame = pd.read_csv(file_path, comment='#')

# Cut first corrupted cycles and last, where the pole is not swinging anymore
# Take care! - Check also that the position is constant

data = data.iloc[2500:-1800, :]

# Take just data for angle
angle = data['angle']
time = data['time']

# Smooth angle measurement
# https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
angle = savgol_filter(angle, 51, 3)

# Convert units to radians:
mean_angle = np.mean(angle)
angle = angle*2*np.pi/4096.0

# Center around 0
mean_angle = np.mean(angle)
angle = angle - mean_angle

# Compute first derivative
angleD = np.gradient(angle, time)
# and smooth it
angleD = savgol_filter(angleD, 51, 3)


# Compute second derivative
angleDD = np.gradient(angleD, time)
# and smooth it
angleDD = savgol_filter(angleDD, 51, 3)

# Plot angle or its derivative
# plt.figure()
# # plt.plot(time, angle/max(angle), label='Angle')
# plt.plot(time, angle*(180/np.pi), label='Angle')
# # plt.plot(time, angleD/max(angleD), label='AngleD')
# # plt.plot(time, angleDD/max(angleDD), label='AngleDD')
# plt.legend()
# plt.ylabel('Angle(-/D/DD)[-]')
# plt.ylabel('Time [s]')
# plt.show()

# Start with small angle approx to get the approx parameters:
time_small = time[-3000:]
angle_small = angle[-3000:]
angleD_small = angleD[-3000:]
angleDD_small = angleDD[-3000:]

xdata_small = np.array([angle_small, angleD_small]).transpose()
# You can also see the effect of neglecting friction
# xdata_small =  np.reshape(angle_small, (len(angle_small),1))
ydata_small = np.reshape(angleDD_small, (len(angleDD_small),1))

reg = LinearRegression().fit(xdata_small, ydata_small)
print()
print('Small angle approx.:')
print(reg.coef_)
print('J_friction = {}'.format( -reg.coef_[:,1]*(k*m*(L**2)) ))

# Compare prediction based on found parameters with ground truth
angleDD_small_predicted = reg.predict(xdata_small)

plt.figure()
plt.plot(time_small, angleDD_small, label='AngleDD')
plt.plot(time_small, angleDD_small_predicted, label='AngleDD_predicted')
plt.legend()
plt.ylabel('AngleDD [$rad/s^2$]')
plt.xlabel('Time [s]')
plt.show()

# We get a = -37.43 (compared with predicted -37.25) and b = -0.167 hence J_friction = 7.5e-4

# Try with big angles
# Function describing the relation between angle derivatives
def func(x, a,b):
    # x[0] is angle
    # x[1] is angleD
    # returns angleDD
    return a*np.sin(x[0])+b*x[1]


xdata = np.array([angle, angleD])
ydata = angleDD
# Initial guess, first taken from theoretical prediction, second from small angle approx
p0 = np.array([-37.25, -0.167])

bounds = ([-np.inf, -np.inf], [-0.0, 0.0])
popt, pcov = curve_fit(func, xdata, ydata, p0=p0, maxfev=1800, bounds=bounds)

print()
print('Big angles calculation (free):')
print(popt)
print('J_friction = {}'.format(-popt[1]*(k*m*(L**2))))
# We get a = -38.11 (compared with predicted -37.25) and b = -0.055 hence J_friction = 2.5e-4

# Now we impose bounds on a to be between theoretical result and the result we got from small angle approx.
bounds = ([-37.43, -np.inf], [-37.25, 0.0])
popt, pcov = curve_fit(func, xdata, ydata, p0=p0, maxfev=1800, bounds=bounds)

print()
print('Big angles calculation (constrained):')
print(popt)
print('J_friction = {}'.format(-popt[1]*(k*m*(L**2))))
# We get a = -37.43 (which is upper bound) and b = -0.054 hence J_friction = 2.4e-4

angleDD_predicted = func(xdata, *popt)

plt.figure()
plt.plot(time, angleDD, label='AngleDD')
plt.plot(time, angleDD_predicted, label='AngleDD_predicted')
plt.legend()
plt.ylabel('AngleDD [$rad/s^2$]')
plt.xlabel('Time [s]')
plt.show()

"""
It seems to me reasonable to take as true a = -37.43, the result of small angle analysis.
For big angle analysis, oscillations with big amplitude probably totally dominate the error,
although they are not necessarily more precise.
Additionally a = -37.43 is closer to theoretically predicted value

However for J_friction I propose 2.5e-4, an approximate result of big angle analysis.
One can argue that for big angles effect of friction is more noticeable,
we get this value also with imposed constrain on "a" from small angle analysis and 
the results for small angles with this value are still good.
It would be good to perform some further analysis with relative error,
so that big and small oscillation can affect the regression in the same way.
It is over the scope of my work for this moment.
"""
"""
"So I tried to answer the question: We want to change k=4/3 to k = 1/3.
By what factor do we need to multiply M
so that the positionDD for each state remains as close to the value it had with k=4/3 as possible?
The effect of the correction is highly state dependant,
however the optimal correction seems to be always roughly M*1.013.
The error between the reference value of positionDD with k=4/3
and M*1.0 and the obtained value positionDD with k=1/3 and M*1.013 can be plotted with this script.
Motivation was that we obtained much better concordance with experimental swinging data (positionDD = 0) for k=1/3
but, the results for positionDD is better with k=4/3"
"""


from CartPole.cartpole_parameters import (
    k, m_cart, m_pole, g, J_fric, M_fric, L
)

import numpy as np

import matplotlib.pyplot as plt

def _cartpole_ode (ca, sa, angleD, positionD, u,
                   k=k, m_cart=m_cart, m_pole=m_pole, g=g, J_fric=J_fric, M_fric=M_fric, L=L):

    """
    Calculates current values of second derivative of angle and position
    from current value of angle and position, and their first derivatives

    :param angle, angleD, position, positionD: Essential state information of cart
    :param u: Force applied on cart in unnormalized range

    :returns: angular acceleration, horizontal acceleration
    """

    # Clockwise rotation is defined as negative
    # force and cart movement to the right are defined as positive
    # g (gravitational acceleration) is positive (absolute value)
    # Checked independently by Marcin and Krishna

    A = (k + 1) * (m_cart + m_pole) - m_pole * (ca ** 2)
    F_fric = - M_fric * positionD  # Force resulting from cart friction, notice that the mass of the cart is not explicitly there
    T_fric = - J_fric * angleD  # Torque resulting from pole friction

    positionDD = (
            (
                    + m_pole * g * sa * ca  # Movement of the cart due to gravity
                    + ((T_fric * ca) / L)  # Movement of the cart due to pend' s friction in the joint
                    + (k + 1) * (
                            - (m_pole * L * (
                                        angleD ** 2) * sa)  # Keeps the Cart-Pole center of mass fixed when pole rotates
                            + F_fric  # Braking of the cart due its friction
                            + u  # Effect of force applied to cart
                    )
            ) / A
    )

    return positionDD


angle = np.linspace(-np.pi, np.pi, 1000)
ca = np.cos(angle)
sa = np.sin(angle)

k = 1.0/3.0

angleD = 0.9
positionD = -.50
u = 0.8

def positionDD_1THIRD(M_factor): return _cartpole_ode(ca, sa, angleD=angleD, positionD=positionD, u=u, k=k, m_cart=M_factor * m_cart)
def positionDD_4THIRDS(): return _cartpole_ode(ca, sa, angleD=angleD, positionD=positionD, u=u, k=k+1.0, m_cart=m_cart)

def SE(M_factor): return (positionDD_4THIRDS()-positionDD_1THIRD(M_factor))**2

# M_min = 0.8
# M_max = 1.3

# difference from 1
# M_min = 0.99
# M_max = 1.015

#Zooned
M_min = 1.01
M_max = 1.02
M_factor = np.linspace(M_min, M_max, 100)


meanSE = []
for m_cart in M_factor:
    meanSE.append(np.sum(SE(m_cart)) / len(angle))

meanSE = np.array(meanSE)

maxSE = []
for m_cart in M_factor:
    maxSE.append(max(SE(m_cart)))

maxSE = np.array(maxSE)


plt.figure()
plt.title('meanSE error')
plt.xlabel('M')
plt.ylabel('meanSE error')
plt.plot(M_factor, meanSE)
plt.show()

plt.figure()
plt.title('maxSE error')
plt.xlabel('M')
plt.ylabel('maxSE error')
plt.plot(M_factor, maxSE)
plt.show()

M_factor_optimal = 1.013
plt.figure()
plt.title('SE for M corrected and M uncorrecred')
plt.xlabel('angle')
plt.ylabel('SquaredError')
plt.plot(angle, SE(1.0), label='Uncorrected')
plt.plot(angle, SE(M_factor_optimal), label='Corrected with {:.3f}'.format(M_factor_optimal))
plt.legend()
plt.show()

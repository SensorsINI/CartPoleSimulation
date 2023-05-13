# -*- coding: utf-8 -*-
"""
Created on Thu May 11 18:07:58 2023

@author: Shreyan Banerjee
"""

"""
This is a linear-quadratic regulator
It assumes that the input relation is u = Q*u_max (no fancy motor model) !
"""

from SI_Toolkit.computation_library import NumpyLibrary, TensorType
import numpy as np
import scipy
import yaml
import os

from CartPole.cartpole_jacobian import cartpole_jacobian
from CartPole.cartpole_model import s0, u_max
from CartPole.state_utilities import (ANGLE_IDX, ANGLED_IDX, POSITION_IDX,
                                      POSITIOND_IDX)
from Control_Toolkit.Controllers import template_controller
from others.globals_and_utils import create_rng

config = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)
actuator_noise = config["cartpole"]["actuator_noise"]


class controller_energy(template_controller):
    _computation_library = NumpyLibrary
    
    def configure(self):
        # From https://github.com/markwmuller/controlpy/blob/master/controlpy/synthesis.py#L8
        """Solve the continuous time LQR controller for a continuous time system.
        A and B are system matrices, describing the systems dynamics:
         dx/dt = A x + B u
        The controller minimizes the infinite horizon quadratic cost function:
         cost = integral (x.T*Q*x + u.T*R*u) dt
        where Q is a positive semidefinite matrix, and R is positive definite matrix.
        Returns K, X, eigVals:
        Returns gain the optimal gain K, the solution matrix X, and the closed loop system eigenvalues.
        The optimal input is then computed as:
         input: u = -K*x
        """
        self.p_Q = actuator_noise
        # ref Bertsekas, p.151
        self.theta = 0
        self.thetaprev = 0
        self.thetadot = 0
        self.thetadotprev = 0
        self.U = 0
        self.K = 0
        self.deltaU = 0

        seed = self.config_controller["seed"]
        self.rng = create_rng(self.__class__.__name__, seed if seed==None else seed*2)

        # Calculate Jacobian around equilibrium
        # Set point around which the Jacobian should be linearized
        # It can be here either pole up (all zeros) or pole down
        s = s0
        s[POSITION_IDX] = 0.0
        s[POSITIOND_IDX] = 0.0
        s[ANGLE_IDX] = 0.0
        s[ANGLED_IDX] = 0.0
        u = 0.0
        
        

        jacobian = cartpole_jacobian(s, u)
        A = jacobian[:, :-1]
        B = np.reshape(jacobian[:, -1], newshape=(4, 1)) * u_max

        # Cost matrices for LQR controller
        self.Q = np.diag(self.config_controller["Q"]) # How much to punish x, v, theta, omega
        self.R = self.config_controller["R"]  # How much to punish Q

        # first, try to solve the ricatti equation
        X = scipy.linalg.solve_continuous_are(A, B, self.Q, self.R)

        # compute the LQR gain
        if np.array(self.R).ndim == 0:
            Ri = 1.0 / self.R
        else:
            Ri = np.linalg.inv(self.R)

        K = np.dot(Ri, (np.dot(B.T, X)))

        eigVals = np.linalg.eigvals(A - np.dot(B, K))

        self.K = K
        self.X = X
        self.eigVals = eigVals

    def controller_reset(self):
        #self.optimizer.optimizer_reset()
        #self.cartpole_trajectory_generator.reset()
        pass

    def step(self, s: np.ndarray, time=None, updated_attributes: "dict[str, TensorType]" = {}):
        self.update_attributes(updated_attributes)
        
        state = np.array(
            [[s[POSITION_IDX] - self.variable_parameters.target_position], [s[POSITIOND_IDX]], [s[ANGLE_IDX]], [s[ANGLED_IDX]]])

        #Q = np.dot(-self.K, state).item()
        
        self.theta = s[ANGLE_IDX]
        self.thetadot = s[ANGLED_IDX]
        
        m = 1
        l = 1
        mc = 1
        g = 9.8
        
        #Modify
        self.K = 0.5 * m*l*l * self.thetadot**2
        self.U = 0.5 * m*g*l*np.cos(self.theta)
        E = self.K+self.U
        
        
        u = ((m*g*abs(np.sin(self.theta))/9.8)*2)*self.thetadot/(abs(self.thetadot+0.01))
        #u = (m*g*abs(np.sin(theta)) - m*l*thetaddot)*2/g
        #c1+=1
        self.deltaU = m*g*l*(1-np.cos(self.theta))
        # if thetadot*theta>0:
        #     pausefactor = -K + deltaU
        #     #c = 1/(1+np.exp(pausefactor*10))
        #c1+=1
        # if c1%500==0:
        #     done = True
        
        if (np.cos(self.theta)-np.cos(self.thetaprev))<0:
            if (self.K+self.deltaU - m*g*l)>=0:
                u=0
        else:
            if (self.K-self.deltaU)>=0:
                u=0
        
        u= (2*u/l)*(m/mc)
        u = u * (np.cos(self.theta)/abs(np.cos(self.theta)))
        u*=6
        #print(theta)
        # if abs(theta)>0.9:
        #     print("Failed")
        # theta sign reversal for stabilization problem
        if abs(self.theta)<0.5:# and self.theta*self.thetadot<0:
            # if abs(self.thetadot)<0.05 and abs(theta)>0.3:
            #     u = 
            u = m*g*l*np.sin(self.theta)*1
            u= (2*u/l)*(m/mc)
            u = -u#*(np.cos(theta)/abs(np.cos(theta)))
            u*=1
            #print(u)
        
        #print(u)
        # if c1 <5:
        #     u = 1
        self.thetaprev = self.theta
        self.thetadotprev = self.thetadot
        self.Kprev = self.K
        # time_step = env.step([u])
        # observation = time_step.observation
        # position = T.Tensor(time_step.observation['position'])
        # velocity = T.Tensor(time_step.observation['velocity'])
        # observation = T.cat((position,velocity),0).reshape(len(position)+len(velocity))
        # costheta = position[1].item()
        # sintheta = position[2].item()
        # theta = math.atan2(sintheta,costheta)
        # thetadot = velocity[1].item()
        #thetadotprev = thetadot
        #thetaprev = theta
        

        # Mod ends
        Q = u
        #Q *= (1 + self.p_Q * float(self.rng.uniform(self.action_low, self.action_high)))

        # Clip Q
        Q = np.clip(Q, -1.0, 1.0, dtype=np.float32)
        return Q
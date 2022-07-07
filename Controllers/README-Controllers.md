# README-Controllers

This folder contains controllers to control the CartPole.
The nets folder contain exemplary neural networks for those controllers which need one
and import them directly (not through predictor class)


List of available controllers with description:
    
    - do-mpc:
        based on do-mpc library, contnuous model, we provide do-mpc library with true equations, it internally integrates it with cvodes
        Example of working parameters: dt=0.2, horizon=10, working parameters from git revision number:

    - do-mpc-discrete:
        Same as do-mpc, just discrete model obtained from continuous with single step Euler stepping

    - lqr:
        linear quadratic regulator controller, our very first well working controller

    - mpc-opti:
        Custom implementation of MPC with Casadi "opti" library

    - mppi:
        A CPU-only implementation of Model Predictive Path Integral Control (Williams et al. 2015). 
        Thousands of randomly perturbed inputs are simulated through the optimization horizon, 
        then averaged by weighted cost, to produce the next control input.

    -cem-tf
        A standard implementation of the cem algorithm. Samples a number of random input sequences from a normal distribution,
        then simulates them and selectes the 'elite' set of random inputs with lowest costs. The sampling distribution
        is fitted to the elite set and the procedure repeated a fixed number of times. 
        In the end the mean of the elite set is used as input.
    
    -cem-naive-grad-tf
        Same as cem, but between selecting the elite set and fitting the distribution, all input sequences in the elite
        set are refined with a single step of gradient descent.

    -dist-adam-resamp2
        Initially samples a set of control sequences, then optimizes them with the adam optimizer projecting control inputs,
        clipping inputs which violate the constraints. For the next time step, the optimizations are warm started with
        the solution from the last one. In regular intervals the only a subset of cheap control sequences are 
        warm started, while the other ones are resampled.

    -mppi-optimze
        First find an initial guess of control sequence with the standard mppi approach. Then optimze it using the adam
        optimizer.
    
        

    
        
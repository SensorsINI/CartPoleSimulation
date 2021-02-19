# README-Controllers

This folder contains controllers to control the CartPole.
The nets folder contain exemplary neural networks for those controllers which need one
and import them directly (not through predictor class)


List of available controllers with description:
    
    - do-mpc:
        based on do-mpc library, contnuous model, we provide do-mpc library with true equations, it internally integrates it with cvodes
        Example of working parameters: dt=0.2, horizon=10, working parameters from git revision number:

    - lqr:
        linear quadratic regulator controller, our very first well working controller

    - controller_rnn_as_mpc(_tf):
        RNN imitating MPC (E2E learning). Version for Pytorch RNN and Tensorflow
        Working in revision:

    
        

    
        
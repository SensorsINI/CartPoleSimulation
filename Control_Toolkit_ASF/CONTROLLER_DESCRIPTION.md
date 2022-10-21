##  Description of Provided Application-Specific Controllers

- `do-mpc`:
    based on do-mpc library, contnuous model, we provide do-mpc library with true equations, it internally integrates it with cvodes
    Example of working parameters: dt=0.2, horizon=10, working parameters from git revision number:

- `do-mpc-discrete`:
    Same as do-mpc, just discrete model obtained from continuous with single step Euler stepping

- `lqr`:
    linear quadratic regulator controller, our very first well working controller

- `mpc-opti`:
    Custom implementation of MPC with Casadi "opti" library

- `mppi-cartpole`:
    A CPU-only implementation of Model Predictive Path Integral Control (Williams et al. 2015).
    This controller is application-specific for the simulated cartpole. 
    Thousands of randomly perturbed inputs are simulated through the optimization horizon, 
    then averaged by weighted cost, to produce the next control input.
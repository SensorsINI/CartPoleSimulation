# Application-specific controllers (Control_Toolkit_ASF)

Cartpole controllers used by the GUI and `run_data_generator.py`. Hyperparameters
live in [config_controllers.yml](config_controllers.yml); optimizers in
[config_optimizers.yml](config_optimizers.yml).

## `lqr`

Linear-quadratic regulator. First well-working baseline.

* Config: `lqr` section in `config_controllers.yml` (`Q`, `R`).

## `pid`

Nested angle + position PID (simulation tuning).

* Config: `pid` section (`P_angle`, `I_angle`, …).

## `mpc`

Model predictive control shell; optimizer selected separately.

* Optimizers: `rpgd`, `rpgd-c`, etc. in `config_optimizers.yml`.
* Cost: `config_cost_function.yml`.
* Predictor: `config_predictors.yml` (`ODE`, neural autoregressive, …).

## `neural-imitator`

Supervised network controller (TensorFlow or exported C evaluator).

* Config: `PATH_TO_MODELS`, `net_name`, `nn_evaluator_mode`, `input_precision`.
* Default in physical-cartpole tree: Exp-29 `Dense-7IN-32H1-32H2-1OUT-1`.
* On-chip Dense-8 / LSTM variants: see physical-cartpole
  [examples/models](https://github.com/SensorsINI/physical-cartpole/tree/master/examples/models).

## `mppi-cartpole`

CPU MPPI (Williams et al. 2015): random rollouts, weighted average control.

* Config: horizon, rollouts, predictor, cost weights in `config_controllers.yml`.

## `do-mpc` / `do-mpc-discrete`

MPC via [do-mpc](https://www.do-mpc.com/) on the true cartpole ODE (continuous
or Euler-discretized).

* Example working dt/horizon noted in source comments; tune in yaml.

## `mpc-opti`

Custom MPC with CasADi `opti`.

## `secloc*` (research)

SecLoc gate wrappers (`secloc`, `secloc-lqr`, `secloc-do-mpc-discrete`, `secloc-c`).
Used on Zedboard / SecLoc branch — not the Development Zybo show default.

* Gate profile: [config_secloc.yml](config_secloc.yml).

## `difflg` / `c` / `embedded`

Legacy / embedded evaluation paths. See `config_controllers.yml` for firmware
file references.

## Adding a controller

1. Subclass or follow `Control_Toolkit/Controllers/template_controller.py`.
2. Add `controller_myname.py` under `Control_Toolkit_ASF/Controllers/`.
3. Register defaults in `config_controllers.yml`.
4. Select in GUI ([config_gui.yml](../config_gui.yml)) or
   [config_data_gen.yml](../config_data_gen.yml).

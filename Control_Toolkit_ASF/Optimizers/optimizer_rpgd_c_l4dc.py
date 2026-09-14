"""RPGD-C teacher for the L4DC 2025 pole-length / pole-mass A-NC.

The C solver does not call the Python cost. It uses the quadratic-dd / energy
kinetic terms in rpgd_c/cartpole_cost.c. This wrapper copies the L4DC 2024
numeric weights into those C fields and turns off vel_penalty_reg so ekp is
closer to the original angleD**2 term (exact match only when the pole is up).

Optimizer knobs match 16 Nov 2024 rpgd-tf: 16 rollouts, 64 outer iterations,
warmup on. See config_optimizers.yml rpgd_c_l4dc.
"""

from Control_Toolkit.Optimizers.optimizer_rpgd_c import optimizer_rpgd_c
from SI_Toolkit.load_and_normalize import load_yaml
import os


class optimizer_rpgd_c_l4dc(optimizer_rpgd_c):

    def _make_config(self, dt):
        cfg = super()._make_config(dt)
        cost_cfg = load_yaml(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"))
        l4 = cost_cfg["CartPole"]["quadratic_boundary_grad_l4dc"]
        # C treats dd as *quadratic* ((x-xt)/(2T))**2. 2024 dd_weight was linear |x-xt|/(2T).
        cfg.dd_quadratic_weight_up = float(l4["dd_weight"])
        cfg.db_weight_up = float(l4["db_weight"])
        cfg.ep_weight_up = float(l4["ep_weight"])
        cfg.ekp_weight_up = float(l4["ekp_weight"])
        cfg.cc_weight_up = float(l4["cc_weight"])
        cfg.vel_penalty_reg = 0.0
        cfg.R = float(l4["R"])
        cfg.permissible_track_fraction = float(l4["permissible_track_fraction"])
        return cfg

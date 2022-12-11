from types import SimpleNamespace
import numpy as np

from others.globals_and_utils import load_or_reload_config_if_modified

(config, _) = load_or_reload_config_if_modified("config.yml")
CARTPOLE_PHYSICAL_CONSTANTS=config.cartpole # take this branch as the constants of the robot

# Parameters of the CartPole
# CARTPOLE_PHYSICAL_CONSTANTS = SimpleNamespace()  # "p" like parameters
# CARTPOLE_PHYSICAL_CONSTANTS.m_pole = config["cartpole"]["m_pole"]
# CARTPOLE_PHYSICAL_CONSTANTS.m_cart = config["cartpole"]["m_cart"]
# CARTPOLE_PHYSICAL_CONSTANTS.L = float(config["cartpole"]["L"].split("/")[0]) / float(config["cartpole"]["L"].split("/")[1])
# CARTPOLE_PHYSICAL_CONSTANTS.u_max = config["cartpole"]["u_max"]
# CARTPOLE_PHYSICAL_CONSTANTS.M_fric = config["cartpole"]["M_fric"]
# CARTPOLE_PHYSICAL_CONSTANTS.J_fric = config["cartpole"]["J_fric"]
# CARTPOLE_PHYSICAL_CONSTANTS.v_max = config["cartpole"]["v_max"]

# cart_length = config["cartpole"]["cart_length"]
# usable_track_length = config["cartpole"]["usable_track_length"]

CARTPOLE_PHYSICAL_CONSTANTS.TrackHalfLength = (CARTPOLE_PHYSICAL_CONSTANTS.usable_track_length - CARTPOLE_PHYSICAL_CONSTANTS.cart_length) / 2.0  # m, effective length, by which cart center can move

# CARTPOLE_PHYSICAL_CONSTANTS.controlDisturbance = config["cartpole"]["controlDisturbance"]
# CARTPOLE_PHYSICAL_CONSTANTS.controlBias = config["cartpole"]["controlBias"]
#
# CARTPOLE_PHYSICAL_CONSTANTS.g = config["cartpole"]["g"]
# TODO why is k computed this way, from "1.0/3.0" as defined in config.yml? It was from 1.0/3.0
# CARTPOLE_PHYSICAL_CONSTANTS.k = float(CARTPOLE_PHYSICAL_CONSTANTS.k.split("/")[0]) / float(CARTPOLE_PHYSICAL_CONSTANTS.k.split("/")[1])


# Export variables as global
def export_globals():
    return (
    np.array(CARTPOLE_PHYSICAL_CONSTANTS.k, dtype=np.float32),
    np.array(CARTPOLE_PHYSICAL_CONSTANTS.m_cart, dtype=np.float32),
    np.array(CARTPOLE_PHYSICAL_CONSTANTS.m_pole, dtype=np.float32),
    np.array(CARTPOLE_PHYSICAL_CONSTANTS.g, dtype=np.float32),
    np.array(CARTPOLE_PHYSICAL_CONSTANTS.J_fric, dtype=np.float32),
    np.array(CARTPOLE_PHYSICAL_CONSTANTS.M_fric, dtype=np.float32),
    np.array(CARTPOLE_PHYSICAL_CONSTANTS.L, dtype=np.float32),
    np.array(CARTPOLE_PHYSICAL_CONSTANTS.v_max, dtype=np.float32),
    np.array(CARTPOLE_PHYSICAL_CONSTANTS.u_max, dtype=np.float32),
    np.array(CARTPOLE_PHYSICAL_CONSTANTS.controlDisturbance, dtype=np.float32),
    np.array(CARTPOLE_PHYSICAL_CONSTANTS.controlBias, dtype=np.float32),
    np.array(CARTPOLE_PHYSICAL_CONSTANTS.TrackHalfLength, dtype=np.float32)
)

k, m_cart, m_pole, g, J_fric, M_fric, L, v_max, u_max, controlDisturbance, controlBias, TrackHalfLength = export_globals()

from types import SimpleNamespace
import numpy as np

from others.globals_and_utils import load_or_reload_config_if_modified

(config, _) = load_or_reload_config_if_modified("config.yml",search_path=['CartPoleSimulation']) # add search path in case run from physical-cartpole
CARTPOLE_PHYSICAL_CONSTANTS=config.cartpole # take this branch as the constants of the robot

CARTPOLE_PHYSICAL_CONSTANTS.TrackHalfLength = (CARTPOLE_PHYSICAL_CONSTANTS.usable_track_length - CARTPOLE_PHYSICAL_CONSTANTS.cart_length) / 2.0  # m, effective length, by which cart center can move

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
    np.array(CARTPOLE_PHYSICAL_CONSTANTS.TrackHalfLength, dtype=np.float32),
    np.array(CARTPOLE_PHYSICAL_CONSTANTS.cart_bounce_factor, dtype=np.float32)
)

k, m_cart, m_pole, g, J_fric, M_fric, L, v_max, u_max, controlDisturbance, controlBias, TrackHalfLength, cart_bounce_factor = export_globals()

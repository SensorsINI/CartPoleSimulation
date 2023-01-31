from types import SimpleNamespace
import numpy as np

from others.globals_and_utils import load_config

config = load_config("config.yml")

# Parameters of the CartPole
P_GLOBALS = SimpleNamespace()  # "p" like parameters
P_GLOBALS.m_pole = config["cartpole"]["m_pole"]
P_GLOBALS.m_cart = config["cartpole"]["m_cart"]
P_GLOBALS.L = float(config["cartpole"]["L"].split("/")[0])/float(config["cartpole"]["L"].split("/")[1])
P_GLOBALS.u_max = config["cartpole"]["u_max"]
P_GLOBALS.M_fric = config["cartpole"]["M_fric"]
P_GLOBALS.J_fric = config["cartpole"]["J_fric"]
P_GLOBALS.v_max = config["cartpole"]["v_max"]

cart_length = config["cartpole"]["cart_length"]
usable_track_length = config["cartpole"]["usable_track_length"]
P_GLOBALS.TrackHalfLength = (usable_track_length-cart_length)/2.0  # m, effective length, by which cart center can move

P_GLOBALS.controlDisturbance = config["cartpole"]["controlDisturbance"]
P_GLOBALS.controlBias = config["cartpole"]["controlBias"]

P_GLOBALS.g = config["cartpole"]["g"]
P_GLOBALS.k = float(config["cartpole"]["k"].split("/")[0])/float(config["cartpole"]["k"].split("/")[1])


# Export variables as global
def export_globals():
    return (
    np.array(P_GLOBALS.k, dtype=np.float32),
    np.array(P_GLOBALS.m_cart, dtype=np.float32),
    np.array(P_GLOBALS.m_pole, dtype=np.float32),
    np.array(P_GLOBALS.g, dtype=np.float32),
    np.array(P_GLOBALS.J_fric, dtype=np.float32),
    np.array(P_GLOBALS.M_fric, dtype=np.float32),
    np.array(P_GLOBALS.L, dtype=np.float32),
    np.array(P_GLOBALS.v_max, dtype=np.float32),
    np.array(P_GLOBALS.u_max, dtype=np.float32),
    np.array(P_GLOBALS.controlDisturbance, dtype=np.float32),
    np.array(P_GLOBALS.controlBias, dtype=np.float32),
    np.array(P_GLOBALS.TrackHalfLength, dtype=np.float32)
)

k, m_cart, m_pole, g, J_fric, M_fric, L, v_max, u_max, controlDisturbance, controlBias, TrackHalfLength = export_globals()

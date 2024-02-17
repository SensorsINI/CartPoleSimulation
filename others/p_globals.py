from SI_Toolkit.computation_library import NumpyLibrary

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
usable_track_length = config["cartpole"]["track_length"]-cart_length
P_GLOBALS.TrackHalfLength = usable_track_length/2.0  # m, effective length, by which cart center can move

P_GLOBALS.controlDisturbance = config["cartpole"]["controlDisturbance"]
P_GLOBALS.controlBias = config["cartpole"]["controlBias"]

P_GLOBALS.g = config["cartpole"]["g"]
P_GLOBALS.k = float(config["cartpole"]["k"].split("/")[0])/float(config["cartpole"]["k"].split("/")[1])


# Export variables as global
def export_parameters(lib=NumpyLibrary):
    dtype = lib.float32

    if 'trainable' in config['cartpole'] and config['cartpole']['trainable'] is True:
        convert = lambda x: lib.to_variable(x, dtype=dtype)
    else:
        convert = lambda x: lib.to_tensor(x, dtype=dtype)

    return (
        convert(P_GLOBALS.k),
        convert(P_GLOBALS.m_cart),
        convert(P_GLOBALS.m_pole),
        convert(P_GLOBALS.g),
        convert(P_GLOBALS.J_fric),
        convert(P_GLOBALS.M_fric),
        convert(P_GLOBALS.L),
        convert(P_GLOBALS.v_max),
        convert(P_GLOBALS.u_max),
        convert(P_GLOBALS.controlDisturbance),
        convert(P_GLOBALS.controlBias),
        convert(P_GLOBALS.TrackHalfLength)
    )


k, m_cart, m_pole, g, J_fric, M_fric, L, v_max, u_max, controlDisturbance, controlBias, TrackHalfLength = export_parameters()

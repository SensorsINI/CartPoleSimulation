from types import SimpleNamespace
from numpy import float32
import yaml
config = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)

# Parameters of the CartPole
P_GLOBALS = SimpleNamespace()  # "p" like parameters
P_GLOBALS.m = config["cartpole"]["m"]
P_GLOBALS.M = config["cartpole"]["M"]
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
P_GLOBALS.sensorNoise = config["cartpole"]["sensorNoise"]

P_GLOBALS.g = config["cartpole"]["g"]
P_GLOBALS.k = float(config["cartpole"]["k"].split("/")[0])/float(config["cartpole"]["k"].split("/")[1])


# Export variables as global
k, M, m, g, J_fric, M_fric, L, v_max, u_max, sensorNoise, controlDisturbance, controlBias, TrackHalfLength = (
    float32(P_GLOBALS.k),
    float32(P_GLOBALS.M),
    float32(P_GLOBALS.m),
    float32(P_GLOBALS.g),
    float32(P_GLOBALS.J_fric),
    float32(P_GLOBALS.M_fric),
    float32(P_GLOBALS.L),
    float32(P_GLOBALS.v_max),
    float32(P_GLOBALS.u_max),
    float32(P_GLOBALS.sensorNoise),
    float32(P_GLOBALS.controlDisturbance),
    float32(P_GLOBALS.controlBias),
    float32(P_GLOBALS.TrackHalfLength)
)

CARTPOLE_EQUATIONS = 'Marcin-Sharpneat'
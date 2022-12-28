from types import SimpleNamespace
import numpy as np


STATE_VARIABLES = np.sort(
    ["angle", "angleD", "angle_cos", "angle_sin", "position", "positionD",]
)

NUM_STATES:int=len(STATE_VARIABLES)

# dict[state_name, index], e.g. STATE_INDICES['angle']=0
STATE_INDICES = {x: np.where(STATE_VARIABLES == x)[0][0] for x in STATE_VARIABLES}

CONTROL_INPUTS = np.sort(["Q"])

CONTROL_INDICES = {x: np.where(CONTROL_INPUTS == x)[0][0] for x in CONTROL_INPUTS}

"""Define indices of values in state statically"""
ANGLE_IDX = STATE_INDICES["angle"].item() # Pole angle in radians. 0 means pole is upright. Clockwise angle rotation is defined as negative.
ANGLED_IDX = STATE_INDICES["angleD"].item() # Angular velocity of pole in rad/s, CCW is positive.
POSITION_IDX = STATE_INDICES["position"].item() # Horizontal position of pole in meters.
POSITIOND_IDX = STATE_INDICES["positionD"].item() # Horizontal velocity of pole in m/s. Cart movement to the right is positive.
ANGLE_COS_IDX = STATE_INDICES["angle_cos"].item() # cos of angle, vertical means 1 and -1 means hanging down
ANGLE_SIN_IDX = STATE_INDICES["angle_sin"].item() # sin of angle, 0 upright and hanging, +1 for leftwards, -1 for rightwards


def create_cartpole_state(state: dict = {}, dtype=None) -> np.ndarray:
    """
    Constructor of cartpole state from named arguments. The order of variables is fixed in STATE_VARIABLES.

    Input parameters are passed as a dict with the following possible keys. Other keys are ignored.
    Unset key-value pairs are initialized to 0.

    :param angle: Pole angle in radians. 0 means pole is upright. Clockwise angle rotation is defined as negative.
    :param angleD: Angular velocity of pole in rad/s, CCW is positive.
    :param position: Horizontal position of pole in meters.
    :param positionD: Horizontal velocity of pole in m/s. Cart movement to the right is positive.

    :returns: A numpy.ndarray with values filled in order set by STATE_VARIABLES
    """
    initial_pole_angle=0 # set to zero to start upright, np.pi to hang down

    state['angle']=initial_pole_angle

    state["angle_cos"] = (
        np.cos(state["angle"]) if "angle" in state.keys() else np.cos(initial_pole_angle)
    )
    state["angle_sin"] = (
        np.sin(state["angle"]) if "angle" in state.keys() else np.sin(initial_pole_angle)
    )

    if dtype is None:
        dtype = np.float32

    s = np.zeros_like(STATE_VARIABLES, dtype=np.float32)
    for i, v in enumerate(STATE_VARIABLES):
        s[i] = state.get(v) if v in state.keys() else s[i]

    return s


# THE FUNCTIONS BELOW ARE POTENTIALLY SLOW!
def cartpole_state_varname_to_index(variable_name: str) -> int:
    return STATE_INDICES[variable_name]


def cartpole_state_index_to_varname(index: int) -> str:
    return STATE_VARIABLES[index]


def cartpole_state_varnames_to_indices(variable_names: list) -> list:
    indices = []
    for variable_name in variable_names:
        indices.append(cartpole_state_varname_to_index(variable_name))
    return indices


def cartpole_state_indices_to_varnames(indices: list) -> list:
    varnames = []
    for index in indices:
        varnames.append(cartpole_state_index_to_varname(index))
    return varnames


def cartpole_state_namespace_to_vector(s_namespace: SimpleNamespace) -> np.ndarray:
    s_array = np.zeros_like(STATE_VARIABLES, dtype=np.float32)
    for a in STATE_VARIABLES:
        s_array[cartpole_state_varname_to_index(a)] = getattr(
            s_namespace, a, s_array[cartpole_state_varname_to_index(a)]
        )
    return s_array


def cartpole_state_vector_to_namespace(s_vector: np.ndarray) -> SimpleNamespace:
    s_namespace = SimpleNamespace()
    for i, a in enumerate(STATE_VARIABLES):
        setattr(s_namespace, a, s_vector[i])
    return s_namespace


# # Test functions
# s = create_cartpole_state(dict(angleD=12.1, angleDD=-33.5, position=2.3, positionD=-19.77, positionDD=3.42))
# s[POSITIOND_IDX] = -14.9
# cartpole_state_index_to_varname(4)

# sn = SimpleNamespace()
# sn.position=23.55
# sn.angleDD=4.11
# sn.eew = -1.22
# q = cartpole_state_namespace_to_vector(sn)
# v = cartpole_state_vector_to_namespace(q)

# print(s)

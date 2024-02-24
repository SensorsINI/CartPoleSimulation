from SI_Toolkit.computation_library import NumpyLibrary
from ruamel.yaml import YAML

from types import SimpleNamespace

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


class CartPoleParameters:
    def __init__(self, lib=NumpyLibrary, get_parameters_from=None):
        self.lib = lib
        if get_parameters_from is None:
            (self.k, self.m_cart, self.m_pole, self.g,
             self.J_fric, self.M_fric, self.L,
             self.v_max, self.u_max, self.controlDisturbance,
             self.controlBias, self.TrackHalfLength) = export_parameters(lib)
        else:
            # Initialize ruamel.yaml object
            yaml = YAML()

            # Load the parameters from a YAML file
            with open(get_parameters_from, 'r') as file:
                parameters = yaml.load(file)
                for key, value in parameters.items():
                    setattr(self, key, value)

    def save_parameters(self, filepath='cartpole_parameters.yml'):
        # Convert SimpleNamespace to a dictionary
        params_dict = {param_name: getattr(self, param_name) for param_name in self.__dict__ if param_name != 'lib'}

        # Initialize ruamel.yaml object
        yaml = YAML()
        yaml.indent(mapping=2, sequence=4, offset=2)

        # Save the dictionary to a YAML file
        with open(filepath, 'w') as file:
            yaml.dump(params_dict, file)

    def export_parameters(self):
        return (self.k, self.m_cart, self.m_pole, self.g,
                self.J_fric, self.M_fric, self.L,
                self.v_max, self.u_max, self.controlDisturbance,
                self.controlBias, self.TrackHalfLength)


k, m_cart, m_pole, g, J_fric, M_fric, L, v_max, u_max, controlDisturbance, controlBias, TrackHalfLength = export_parameters()

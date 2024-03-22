from SI_Toolkit.computation_library import NumpyLibrary
from ruamel.yaml import YAML

from others.globals_and_utils import load_config


class CartPoleParameters:
    def __init__(self, lib=NumpyLibrary, get_parameters_from=None):
        self.lib = lib
        if get_parameters_from is None:
            get_parameters_from = "cartpole_physical_parameters.yml"

        parameters = load_config(get_parameters_from)['cartpole']

        for key, value in parameters.items():
            if key == 'L':
                value = float(value.split("/")[0])/float(value.split("/")[1])
            elif key == 'k':
                value = float(value.split("/")[0])/float(value.split("/")[1])
            if key in ['k', 'm_cart', 'm_pole', 'g', 'J_fric', 'M_fric', 'L', 'v_max', 'u_max',
                       'controlDisturbance', 'controlBias', 'TrackHalfLength']:
                value = lib.to_tensor(value, dtype=lib.float32)
            setattr(self, key, value)
            setattr(self, 'TrackHalfLength', lib.to_tensor((parameters['track_length']-parameters['cart_length'])/2.0, dtype=lib.float32))

    def save_parameters(self, filepath='cartpole_parameters.yml'):
        # Convert SimpleNamespace to a dictionary
        params_dict = {param_name: getattr(self, param_name) for param_name in self.__dict__ if param_name != 'lib'}

        # Initialize ruamel.yaml object
        yaml = YAML()
        yaml.indent(mapping=2, sequence=4, offset=2)

        # Save the dictionary to a YAML file
        with open(filepath, 'w') as file:
            yaml.dump(params_dict, file)

    def export_parameters(self, lib=None):

        if lib is None:
            convert = lambda x: x
        else:
            dtype = lib.float32
            convert = lambda x: lib.to_tensor(x, dtype=dtype)


        return (
            convert(self.k),
            convert(self.m_cart),
            convert(self.m_pole),
            convert(self.g),
            convert(self.J_fric),
            convert(self.M_fric),
            convert(self.L),
            convert(self.v_max),
            convert(self.u_max),
            convert(self.controlDisturbance),
            convert(self.controlBias),
            convert(self.TrackHalfLength)
        )


CP_PARAMETERS_DEFAULT = CartPoleParameters()
(k, m_cart, m_pole, g, J_fric, M_fric, L, v_max, u_max,
 controlDisturbance, controlBias, TrackHalfLength) = CP_PARAMETERS_DEFAULT.export_parameters()

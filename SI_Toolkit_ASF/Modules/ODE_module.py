import os.path

import tensorflow as tf
from ruamel.yaml import YAML

from SI_Toolkit.Functions.General.Normalising import get_normalization_function, get_denormalization_function
from SI_Toolkit.Functions.General.Initialization import get_norm_info_for_net
from SI_Toolkit.computation_library import TensorFlowLibrary
from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper

# This is a custom keras model for the ODE model of the car dynamics
# It is not a real network, but it can calculate the ODE with neuw paramteres without recompiling the TensorFlow Graph

predictor_specification = "ODE"
dt = 0.02
trainable_params = ['u_max']

class ODEModel(tf.keras.Model):

    # 1st important function for a "fake network" keras model
    def __init__(self, horizon, batch_size, net_info, name=None, **kwargs):
        super().__init__(**kwargs)

        self.lib = TensorFlowLibrary

        self.batch_size = batch_size
        self.horizon = horizon

        self.predictor = PredictorWrapper()
        self.predictor.update_predictor_config_from_specification(predictor_specification=predictor_specification)
        self.predictor.configure_with_compilation(batch_size=self.batch_size, horizon=self.horizon, dt=dt)

        self.cartpole_params_tf = {}
        for name, param in self.predictor.predictor.params.__dict__.items():
            if trainable_params == 'all':
                trainable = True
            else:
                trainable = name in trainable_params
            self.cartpole_params_tf[name] = tf.Variable(param, name=name, trainable=trainable, dtype=tf.float32)
        for name, var in self.cartpole_params_tf.items():
            setattr(self.predictor.predictor.params, name, var)

    # 2nt important function for a "fake network" keras model
    # Will be called in every training/predition step
    def call(self, x, training=None, mask=None):
        Q = x[:, :, 0:1]
        s = x[:, 0, 1:]
        output = self.predictor.predict_core(s, Q)
        return output

    def save(self, filepath, overwrite=True, save_format=None, **kwargs):
        # Dictionary to hold parameter names and values
        path = filepath[:-len('.keras')]
        params_dict = {}
        for name, var in self.predictor.predictor.params.__dict__.items():
            # For a single scalar value, convert to Python float
            if var.numpy().size == 1:
                params_dict[name] = var.numpy().item()
            # For arrays, convert to list
            else:
                params_dict[name] = var.numpy().tolist()


        yaml = YAML()
        yaml.indent(mapping=2, sequence=4, offset=2)
        with open(path + '.yaml', 'w') as f:
            yaml.dump(params_dict, f)


class CartpoleModel(ODEModel):
    def __init__(self, horizon, batch_size, net_info, name=None, **kwargs):
        super().__init__(horizon, batch_size, net_info, name=None, **kwargs)
import os.path

import tensorflow as tf
from ruamel.yaml import YAML

from SI_Toolkit.Functions.General.Normalising import get_normalization_function, get_denormalization_function
from SI_Toolkit.Functions.General.Initialization import get_norm_info_for_net
from SI_Toolkit.computation_library import TensorFlowLibrary
from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper

# Specification and settings
predictor_specification = "ODE_with_NeuralModel_correction"
dt = 0.01
trainable_params = ['u_max']

# Allow the user to choose what learns:
#   'ode' ........ only physical parameters
#   'nn' ......... only neural-network residual
#   'both' ....... both groups at once
train_mode = 'ode'     # ← set to 'nn' or 'both' as desired


class ODEModel(tf.keras.Model):
    """Custom Keras model combining an ODE integrator with a neural-network correction."""
    def __init__(self, horizon, batch_size, net_info, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.lib = TensorFlowLibrary()
        self.batch_size = batch_size
        self.horizon = horizon

        # 1) Instantiate predictor and mark residual net trainability *before* graph compilation
        self.predictor = PredictorWrapper()
        self.predictor.update_predictor_config_from_specification(
            predictor_specification=predictor_specification
        )

        # 2) Compile the combined ODE + NN graph
        self.predictor.configure_with_compilation(
            batch_size=self.batch_size,
            horizon=self.horizon,
            dt=dt
        )

        # Extract the residual network
        try:
            cpe = self.predictor.predictor.next_step_predictor.cpe
            residual_net_evaluator = getattr(cpe, 'second_derivatives_neural_model', None)
            self.residual_net = residual_net_evaluator.net
        except AttributeError:
            self.residual_net = None

        if train_mode in ('nn', 'both'):
            if self.residual_net is None:
                raise ValueError(
                    "The chosen train_mode requires a residual network, but none was found."
                )
            self.residual_net.trainable = True
        else:
            if self.residual_net is not None:
                self.residual_net.trainable = False

        # 3) Wrap physical parameters as tf.Variables with appropriate trainability
        self.cartpole_params_tf = {}
        for name, param in vars(self.predictor.predictor.params).items():
            if name in trainable_params:
                is_ode_param = (train_mode in ('ode', 'both'))
                trainable = is_ode_param
                try:
                    init_val = param.numpy()
                except Exception:
                    init_val = param
                shape = getattr(init_val, 'shape', ())
                var = self.add_weight(
                    name=name,
                    shape=shape,
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(init_val),
                    trainable=trainable
                )
                self.cartpole_params_tf[name] = var
                setattr(self.predictor.predictor.params, name, var)

        # 4) Register residual network weights so Keras will optimize them
        if self.residual_net is not None and train_mode in ('nn', 'both'):
            # Attempt to gather trainable variables from residual_net
            if hasattr(self.residual_net, 'trainable_variables'):
                var_list = self.residual_net.trainable_variables
            elif hasattr(self.residual_net, 'variables'):
                var_list = [v for v in self.residual_net.variables if getattr(v, 'trainable', True)]
            elif hasattr(self.residual_net, 'weights'):
                var_list = [w for w in self.residual_net.weights if getattr(w, 'trainable', True)]
            else:
                raise ValueError(
                    "Residual network has no accessible variables: cannot register trainable weights."
                )
            for v in var_list:
                self._trainable_weights.append(v)

    def call(self, x, training=None, mask=None):
        """
        Forward pass: predict_core encapsulates ODE integration
        and neural correction in one graph.
        """
        Q = x[:, :, 0:1]
        s = x[:, 0, 1:]
        return self.predictor.predict_core(s, Q)

    def save(self, filepath, overwrite=True, save_format=None, **kwargs):
        # First save the Keras model
        super().save(filepath, overwrite=overwrite, save_format=save_format, **kwargs)

        # Extract physical parameters to YAML
        path = filepath[:-len('.keras')]
        params_dict = {}
        for name, var in vars(self.predictor.predictor.params).items():
            if name in ['J_fric', 'L', 'm_cart', 'M_fric',
                        'TrackHalfLength', 'g', 'k', 'm_pole', 'u_max']:
                val = var.numpy()
                if getattr(val, 'size', 1) == 1:
                    params_dict[name] = float(val)
                else:
                    params_dict[name] = val.tolist()

        yaml = YAML()
        yaml.indent(mapping=2, sequence=4, offset=2)
        with open(path + '.yaml', 'w') as f:
            yaml.dump(params_dict, f)


class CartpoleModel(ODEModel):
    """Concrete instantiation of ODEModel for the cart-pole system."""
    def __init__(self, horizon, batch_size, net_info, name=None, **kwargs):
        super().__init__(horizon, batch_size, net_info, name=name, **kwargs)

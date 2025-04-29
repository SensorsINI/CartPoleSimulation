# ODE_module.py

import os

import tensorflow as tf
from ruamel.yaml import YAML

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
train_mode = 'nn'     # ← set to 'nn' or 'both' as desired


class ODEModel(tf.keras.Model):
    """Custom Keras model combining an ODE integrator with a neural-network correction."""
    def __init__(self, horizon, batch_size, net_info, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.lib = TensorFlowLibrary()
        self.batch_size = batch_size
        self.horizon = horizon
        self._network_folder = None

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
        # — First: dump ODE parameters into YAML (always in default folder) —
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

        # — Then: if we're only training ODE, save the full model here… —
        if train_mode == 'ode':
            super().save(filepath, overwrite=overwrite,
                         save_format=save_format, **kwargs)

        # — …otherwise we only save the NEURAL net into its own folder —
        else:
            parent = os.path.dirname(filepath)
            base   = os.path.basename(path)  # e.g. "Dense-8IN-…-1OUT-0"

            # — compute the network‐folder name only on first save —
            if self._network_folder is None:
                self._network_folder = self._next_network_folder(parent, base)
            newdir = self._network_folder
            full   = os.path.join(parent, newdir)
            os.makedirs(full, exist_ok=True)

            # save ONLY the residual network to avoid ODE duplication
            net_fp = os.path.join(full, newdir + '.keras')
            if self.residual_net is None:
                raise RuntimeError("No residual_net to save, but train_mode!='ode'")
            else:
                print(f"Saving residual network to {net_fp}")
            self.residual_net.save(
                net_fp,
                overwrite=overwrite,
                save_format=save_format,
                **kwargs
            )

            # — NEW: also save weights as a TensorFlow checkpoint (.ckpt) —
            weights_fp = os.path.join(full, newdir + '.ckpt')
            # save_weights will produce .data-00000-of-00001 and .index files
            print(f"Saving residual network weights to {weights_fp}")
            self.residual_net.save_weights(
                weights_fp,
                overwrite=overwrite
            )

    @property
    def layers(self):
        """
        Expose both the ODEModel’s own layers and the nested residual_net’s layers
        so that code doing `for layer in model.layers:` will see the network layers.
        """
        base = super().layers
        if self.residual_net is not None:
            return base + list(self.residual_net.layers)
        return base

    def get_config(self):
        """
        Needed so that `model.save()` won’t warn about non-serializable __init__ args.
        We only serialize the numeric bits; net_info is omitted since we never
        reload this full model from disk.
        """
        base = super().get_config()
        base.update({
            'horizon': self.horizon,
            'batch_size': self.batch_size,
            # note: net_info is intentionally NOT serialized
        })
        return base

    @staticmethod
    def _next_network_folder(parent_dir: str, base_name: str) -> str:
        """
        Scan parent_dir for directories named prefix-<number>, where
        base_name == prefix-<initial_idx>.  Return the new subfolder name
        prefix-<next_idx>.
        """
        prefix, initial_idx = base_name.rsplit('-', 1)
        try:
            init_idx = int(initial_idx)
        except ValueError:
            init_idx = 0

        existing = []
        for name in os.listdir(parent_dir):
            full = os.path.join(parent_dir, name)
            if os.path.isdir(full) and name.startswith(prefix + '-'):
                try:
                    existing.append(int(name.rsplit('-', 1)[1]))
                except ValueError:
                    pass

        if existing:
            next_idx = max(existing) + 1
        else:
            next_idx = init_idx

        return f"{prefix}-{next_idx}"


class CartpoleModel(ODEModel):
    """Concrete instantiation of ODEModel for the cart-pole system."""
    def __init__(self, horizon, batch_size, net_info, name=None, **kwargs):
        super().__init__(horizon, batch_size, net_info, name=name, **kwargs)

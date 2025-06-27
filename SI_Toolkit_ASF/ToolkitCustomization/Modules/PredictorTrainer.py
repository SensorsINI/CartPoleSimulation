# ODE_module.py

import os

import tensorflow as tf
from ruamel.yaml import YAML

from SI_Toolkit.computation_library import TensorFlowLibrary
from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper
from SI_Toolkit_ASF.ToolkitCustomization.Modules.NetworkManager import NetworkManager

# Specification and settings
predictor_specification = "ODE_with_NeuralModel_correction_v2"
dt = 0.01
trainable_params = ['u_max']

# Allow the user to choose what learns:
#   'ode' ........ only physical parameters
#   'nn' ......... only neural-network
#   'both' ....... both groups at once
train_mode = 'nn'     # ← set to 'nn' or 'both' as desired



class PredictorTrainer(tf.keras.Model):
    """Custom Keras model combining an ODE integrator with a neural‑network correction."""

    def __init__(self, time_series_len, batch_size, model_info, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.lib = TensorFlowLibrary()
        self.batch_size = batch_size
        self.time_series_len = time_series_len
        self._network_folder = None

        # Persist training‑related meta‑information
        self.model_info = model_info                       # Holds wash‑out configuration

        # 1) Instantiate predictor and mark net trainability *before* graph compilation
        self.predictor = PredictorWrapper()
        self.predictor.update_predictor_config_from_specification(
            predictor_specification=predictor_specification
        )

        # 2) Compile the combined ODE + NN graph
        self.predictor.configure_with_compilation(
            batch_size=self.batch_size,
            horizon=self.model_info.post_wash_out_len,
            dt=dt
        )

        # Extract the network
        try:
            cpe = self.predictor.predictor.next_step_predictor.cpe
            net_evaluator = getattr(cpe, 'second_derivatives_neural_model', None)
            self.net_manager = NetworkManager(net_evaluator)
        except AttributeError:
            self.net_manager = None


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

        self.net_manager.register_trainable_weights(self, train_mode)

    def call(self, x, training=None, mask=None):
        """
        Forward pass: perform wash-out warm-up via a TensorArray loop,
        then post-wash-out integration, drop the seed, and concat.
        """
        # total time steps in this batch
        w_len     = self.model_info.wash_out_len
        p_len     = self.model_info.post_wash_out_len

        # --- Warm-up loop with TensorArray to avoid scope issues ---
        # get one step to infer dtype & shape
        Q0 = x[:, 0, 0:1]
        s0 = x[:, 0, 1:]
        y0 = self.predictor.predict_next_step(s0, Q0)

        ta = tf.TensorArray(dtype=y0.dtype, size=w_len)
        def _cond(i, ta):
            return i < w_len

        def _body(i, ta):
            Q_curr = x[:, i, 0:1]
            s_curr = x[:, i, 1:]
            y_next = self.predictor.predict_next_step(s_curr, Q_curr)
            ta = ta.write(i, y_next)
            return i + 1, ta

        _, ta = tf.while_loop(_cond, _body, (0, ta))
        # ta.stack() shape = (w_len, batch, output_dim)
        y_warm = ta.stack()
        # reorder to (batch, w_len, output_dim)
        y_warm = tf.transpose(y_warm, perm=[1, 0, 2])

        # --- Post-wash-out prediction ---
        Q_post = x[:, w_len : w_len + p_len, 0:1]
        s_curr = x[:, w_len, 1:]
        y_pred = self.predictor.predict_core(s_curr, Q_post)
        # y_pred shape = (batch, p_len+1, output_dim); drop the seed
        y_post = y_pred[:, 1:, :]

        # concat along time → (batch, total_len, output_dim)
        return tf.concat([y_warm, y_post], axis=1)


    def train_step(self, data):
        """
        Implements wash-out pre-integration followed by supervised learning
        over a user-specified post-wash-out horizon,
        reusing call() so train vs. eval logic is identical.
        """
        # Unpack (x, y)
        x, y = data

        with tf.GradientTape() as tape:
            # reuse call for both warm-up & post-wash-out
            y_pred = self(x, training=True)

            loss = self.compiled_loss(
                y,
                y_pred,
                regularization_losses=self.losses
            )

        # backprop
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # update metrics
        self.compiled_metrics.update_state(y, y_pred)
        results = {m.name: m.result() for m in self.metrics}
        results["loss"] = loss
        return results

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
        else:

            self.net_manager.save_net(
                overwrite=overwrite,
                save_format=save_format,
                **kwargs,
            )

    @property
    def layers(self):
        """
        Expose both the model’s own layers *and* those of the embedded NN.
        """
        if self.net_manager is not None:
            return super().layers + self.net_manager.layers
        return super().layers

    def get_config(self):
        """
        Needed so that `model.save()` won’t warn about non-serializable __init__ args.
        We only serialize the numeric bits;
        """
        base = super().get_config()
        base.update({
            'horizon': self.horizon,
            'batch_size': self.batch_size,
        })
        return base


class CartpoleModel(PredictorTrainer):
    """Concrete instantiation of ODEModel for the cart-pole system."""
    def __init__(self, horizon, batch_size, net_info, name=None, **kwargs):
        super().__init__(horizon, batch_size, model_info=net_info, name=name, **kwargs)

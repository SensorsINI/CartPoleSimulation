import os
import shutil
from SI_Toolkit.Functions.General.Initialization import create_full_name


def copy_file(src, dst, filename):
    """
    Copies *filename* from *src* to *dst*.
    Creates the destination folder if it does not yet exist.
    """
    if not os.path.exists(dst):
        os.makedirs(dst)
    shutil.copy(os.path.join(src, filename), os.path.join(dst, filename))


class NetworkManager:
    """
    Encapsulates all neural‑network plumbing:

      1. Extract the network from any *net_evaluator* that exposes
         `.net` and `.net_info`.
      2. Create a dedicated folder with the correct name, including
         normalization vectors and config YAMLs.
      3. Provide helpers for saving the network and its weights.
      4. Register trainable variables with a parent `tf.keras.Model`.
      5. Expose the network’s layers for seamless iteration.

    Usage (inside a Keras model):

        net_mgr = NetworkManager(net_evaluator)
        net_mgr.register_trainable_weights(self, train_mode)
        ...
        net_mgr.save_net(folder_path, net_full_name, ...)
    """

    # ------------------------------------------------------------------ #
    # Construction & folder materialisation
    # ------------------------------------------------------------------ #
    def __init__(self, net_evaluator):
        """
        Parameters
        ----------
        net_evaluator : object | None
            An object exposing attributes `.net` (tf.keras.Model) **and**
            `.net_info` (namespace with path/name meta‑data).  If *None*
            or these attributes are missing, the manager becomes a no‑op.
        """
        # Gracefully handle “no‑network” situations (pure ODE models)
        try:
            self.net = net_evaluator.net
            self.net_info = net_evaluator.net_info
        except AttributeError:
            self.net = None
            self.net_info = None
            return

        # Ensure a valid full name + target folders before any I/O
        create_full_name(self.net_info, self.net_info.path_to_models)

        # Copy normalisation files & configs into a fresh folder
        self._create_folder()

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #
    def _create_folder(self):
        """
        Populate the new network directory with:
          * normalisation vectors (.csv)
          * YAML config files
          * a descriptor .txt renamed to use the new net name
        """
        base = self.net_info
        dst_folder = os.path.join(base.path_to_models, base.net_full_name)

        parent_folder = os.path.join(base.path_to_models, base.parent_net_name)
        files_to_copy = [
            os.path.basename(base.path_to_normalization_info),
            "normalization_vec_a.csv",
            "normalization_vec_b.csv",
            "denormalization_vec_A.csv",
            "denormalization_vec_B.csv",
            f"{base.parent_net_name}.txt",
        ]
        for fname in files_to_copy:
            copy_file(parent_folder, dst_folder, fname)

        # — Rewrite the descriptor to reflect the *new* network name —
        old_txt = os.path.join(dst_folder, f"{base.parent_net_name}.txt")
        new_txt = os.path.join(dst_folder, f"{base.net_full_name}.txt")
        with open(old_txt, "r") as f:
            content = f.read().replace(base.parent_net_name, base.net_full_name)
        with open(new_txt, "w") as f:
            f.write(content)
        os.remove(old_txt)

        copy_file('./SI_Toolkit_ASF/', dst_folder, 'config_predictors.yml')
        copy_file('./SI_Toolkit_ASF/', dst_folder, 'config_training.yml')

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def save_net(self, overwrite, save_format, **kwargs):
        """
        Persist *only* the neural net (avoids duplicating the ODE model).

        Produces
          • «net_full_name».keras   (full model)
          • «net_full_name».ckpt    (weights checkpoint)
        """

        folder_path = os.path.join(self.net_info.path_to_models, self.net_info.net_full_name)
        net_full_name = self.net_info.net_full_name

        if self.net is None:
            raise RuntimeError("No net to save, but train_mode!='ode'")

        # 1) Full model
        fp_model = os.path.join(folder_path, f"{net_full_name}.keras")
        print(f"Saving network to {fp_model}")
        self.net.save(fp_model, overwrite=overwrite, save_format=save_format, **kwargs)

        # 2) Weights checkpoint
        fp_ckpt = os.path.join(folder_path, f"{net_full_name}.ckpt")
        print(f"Saving network weights to {fp_ckpt}")
        self.net.save_weights(fp_ckpt, overwrite=overwrite)

    def register_trainable_weights(self, parent_model, train_mode):
        """
        Adds the network’s trainable tensors to *parent_model* so that
        a single optimiser step updates both ODE parameters and the NN.

        Parameters
        ----------
        parent_model : tf.keras.Model
            The model whose `trainable_variables` / `_trainable_weights`
            list is to be augmented.
        train_mode : {'ode', 'nn', 'both'}
            Governs whether the NN is trained or frozen.
        """
        if self.net is None:      # nothing to do
            return

        if train_mode in ("nn", "both"):
            self.net.trainable = True
            # Prefer `.trainable_variables`, fall back as needed
            if hasattr(self.net, "trainable_variables"):
                vars_list = self.net.trainable_variables
            elif hasattr(self.net, "variables"):
                vars_list = [v for v in self.net.variables if getattr(v, "trainable", True)]
            else:
                vars_list = [w for w in self.net.weights if getattr(w, "trainable", True)]

            # Keras will now “see” the network’s variables
            for v in vars_list:
                parent_model._trainable_weights.append(v)
        else:
            self.net.trainable = False

    @property
    def layers(self):
        """
        All layers of the managed network (or empty list if absent).
        """
        return [] if self.net is None else list(self.net.layers)

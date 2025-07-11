from SI_Toolkit.Functions.FunctionalDict import FunctionalDict
from Control_Toolkit_ASF.Controllers.difflg_controller.difflg_controller import DiffLogicGateController
from SI_Toolkit.computation_library import TensorType, NumpyLibrary

import numpy as np

from Control_Toolkit.Controllers import template_controller

try:
    from SI_Toolkit_ASF.ToolkitCustomization.predictors_customization import STATE_INDICES
except ModuleNotFoundError:
    print("SI_Toolkit_ASF not yet created")


class controller_difflg(template_controller):
    _computation_library = NumpyLibrary()

    def configure(self):

        self.net_evaluator = DiffLogicGateController()
        self.net_evaluator.load_model()
        self.inputs = ['angleD', 'angle_cos', 'angle_sin', 'position', 'positionD', 'target_equilibrium', 'target_position']

        self.clip_output = self.config_controller.get("clip_output", False)

        self.input_at_input = self.config_controller["input_at_input"]

        # Prepare input mapping
        self.input_mapping = self._create_input_mapping()

        print('Configured DiffLogicGateController.')

    def _create_input_mapping(self):
        """
        Creates a mapping for network inputs to their sources ('state' or 'variable_parameters').

        Returns:
            Dict[str, str]: A dictionary mapping input keys to their sources.
        """
        mapping = {}
        for key in self.inputs:
            if key in STATE_INDICES:
                mapping[key] = 'state'
            else:
                mapping[key] = 'variable_parameters'
        return mapping

    def step(self, s: np.ndarray, time=None, updated_attributes: "dict[str, TensorType]" = {}):

        if self.input_at_input:
            net_input = s
        else:
            self.update_attributes(updated_attributes)
            net_input = self._compose_network_input(s)

        net_input = net_input.astype(np.float32)  # Ensure input is float32 for the model
        Q = self.net_evaluator.predict(net_input)

        if self.clip_output:
            Q = np.clip(Q, -1.0, 1.0)  # Ensure Q is within the range [-1, 1]

        return Q

    def _compose_network_input(self, state: np.ndarray) -> np.ndarray:
        """
        Composes the network input vector from state and variable parameters.

        Args:
            state (np.ndarray): Current state array.

        Returns:
            np.ndarray: Composed network input vector.
        """
        input_vector = []

        for key, source in self.input_mapping.items():
            if source == 'state':
                idx = STATE_INDICES[key]
                input_vector.append(state[idx])
            elif source == 'variable_parameters':
                try:
                    value = getattr(self.variable_parameters, key)
                    input_vector.append(value)
                except AttributeError:
                    raise ValueError(f"Variable parameter '{key}' not found in 'variable_parameters'.")
            else:
                raise ValueError(f"Unknown source '{source}' for input '{key}'.")

        # Convert to numpy array and ensure correct shape
        net_input = np.array(input_vector).reshape(-1)  # Assuming batch_size=1
        return net_input

    def controller_reset(self):
        self.configure()
import os

from SI_Toolkit.load_and_normalize import load_yaml
from SI_Toolkit.General.variable_parameters import VariableParameters

from Control_Toolkit.Cost_Functions.cost_function_wrapper import CostFunctionWrapper


class MPC_CostFunction:

    def __init__(self, lib, initial_environment_attributes):
        self.lib = lib

        self.action_previous = lib.to_variable((0.0,), lib.float32)

        config_controllers = load_yaml(os.path.join("Control_Toolkit_ASF", "config_controllers.yml"))
        cost_function_specification = dict(config_controllers['mpc']).get("cost_function_specification", None)
        self.cost_function = CostFunctionWrapper()

        initial_environment_attributes = {key: lib.to_variable(value, lib.float32) for key, value in
                                          initial_environment_attributes.items()}

        self.variable_parameters = VariableParameters(lib)
        self.variable_parameters.set_attributes(initial_environment_attributes)

        self.cost_function.configure(
            batch_size=1,
            horizon=1,
            variable_parameters=self.variable_parameters,
            environment_name='CartPole',
            computation_library=lib,
            cost_function_specification=cost_function_specification
        )

    def get_cost(self, state, action, environment_attributes=None):

        if environment_attributes is not None:
            self.variable_parameters.update_attributes(environment_attributes)

        stage_cost = self.cost_function.get_summed_stage_cost(
                state[self.lib.newaxis, self.lib.newaxis, :],
                action[self.lib.newaxis, self.lib.newaxis, :],
                self.action_previous[self.lib.newaxis, :])

        self.lib.assign(self.action_previous, action)

        return stage_cost

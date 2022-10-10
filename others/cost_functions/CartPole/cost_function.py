from typing import Union
import tensorflow as tf
from Control_Toolkit.others.environment import EnvironmentBatched, NumpyLibrary, TensorFlowLibrary


class cartpole_cost_function:
    def __init__(self, environment: EnvironmentBatched) -> None:
        self.environment = environment
        if self.environment.lib == TensorFlowLibrary:
            self.target_position_attribute = "target_position_tf"
            self.target_equilibrium_attribute = "target_equilibrium_tf"
        elif self.environment.lib == NumpyLibrary:
            self.target_position_attribute = "target_position"
            self.target_equilibrium_attribute = "target_equilibrium"
        else:
            raise ValueError(
                "Currently, this cost function only supports environment written in TensorFlow or NumPy (not PyTorch etc.)"
            )

    @property
    def target_position(self) -> Union[float, tf.Variable]:
        return getattr(self.environment, self.target_position_attribute)

    @property
    def target_equilibrium(self) -> Union[float, tf.Variable]:
        return getattr(self.environment, self.target_equilibrium_attribute)

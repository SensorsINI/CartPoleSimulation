from typing import Union
import tensorflow as tf
from Control_Toolkit.others.environment import NumpyLibrary, TensorFlowLibrary


class cartpole_cost_function:
    def __init__(self, environment) -> None:
        self.env_mock = environment
        if self.env_mock.lib == TensorFlowLibrary:
            self.target_position_attribute = "target_position_tf"
        elif self.env_mock.lib == NumpyLibrary:
            self.target_position_attribute = "target_position"
        else:
            raise ValueError(
                "Currently, this cost function only supports environment written in TensorFlow or NumPy (not PyTorch etc.)"
            )

    @property
    def target_position(self) -> Union[float, tf.Variable]:
        return getattr(self.env_mock, self.target_position_attribute)

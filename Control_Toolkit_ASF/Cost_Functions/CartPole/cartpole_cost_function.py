from typing import Union

import numpy as np
import tensorflow as tf
import torch
from Control_Toolkit.others.environment import (
    ComputationLibrary,
    NumpyLibrary,
    PyTorchLibrary,
    TensorFlowLibrary,
)
from Control_Toolkit_ASF.Cost_Functions import cost_function_base


class cartpole_cost_function(cost_function_base):
    def __init__(self, ComputationLib: "type[ComputationLibrary]", **kwargs) -> None:
        self.set_computation_library(ComputationLib)
        if self.lib == TensorFlowLibrary:
            self._target_position = tf.Variable(0.0, dtype=tf.float32)
            self._target_equilibrium = tf.Variable(1.0, dtype=tf.float32)
        elif self.lib == NumpyLibrary:
            self._target_position = np.array(0.0, dtype=np.float32)
            self._target_equilibrium = np.array(1.0, dtype=np.float32)
        elif self.lib == PyTorchLibrary:
            self._target_position = torch.tensor(0.0, dtype=torch.float32)
            self._target_equilibrium = torch.tensor(1.0, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown computation library {self.lib}")

    @property
    def target_position(self) -> Union[float, tf.Variable]:
        return self._target_position

    @target_position.setter
    def target_position(self, v):
        self.lib.assign(self._target_position, v)

    @property
    def target_equilibrium(self) -> Union[float, tf.Variable]:
        return self._target_equilibrium

    @target_equilibrium.setter
    def target_equilibrium(self, v):
        self.lib.assign(self._target_equilibrium, v)

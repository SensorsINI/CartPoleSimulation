import numpy as np

from abc import ABC, abstractmethod


"""
Convention for estimators used with cartpole. An estimator must:
1. Be in Estimators folder
2. Have a name starting with "estimator_"
3. The name of the estimator class must be the same as the name of the file.
4. It must have __init__ and step methods

We recommend you derive it from the provided template.
See the provided examples of controllers to gain more insight.
"""


class template_estimator(ABC):

    def __init__(self):
        pass
    
    @abstractmethod
    def step(self, s: np.ndarray, time=None):
        s_estimated = None  # This line is not obligatory. ;-) Just to indicate that s_estimated must me defined and returned
        pass
        return s_estimated  # estimated state of the system


    # Optionally: A method called after an experiment.
    # May be used to print some statistics about estimator performance (e.g. variance change)
    def estimator_report(self):
        raise NotImplementedError

    # Optionally: reset the estimator after an experiment
    # May be useful for stateful estimators, like these containing RNN,
    # To reload the hidden states e.g. if the controller went unstable in the previous run.
    # It is called after an experiment,
    # but only if the estimator is supposed to be reused without reloading (e.g. in GUI)
    def controller_reset(self):
        raise NotImplementedError

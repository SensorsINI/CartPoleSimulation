import os
from typing import Optional
from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper

import numpy as np
import yaml
from Control_Toolkit.Controllers import template_controller
from Control_Toolkit.Cost_Functions.cost_function_wrapper import CostFunctionWrapper

from Control_Toolkit.Optimizers import template_optimizer
from SI_Toolkit.computation_library import TensorType
from Control_Toolkit.others.globals_and_utils import get_logger, import_optimizer_by_name

from torch import inference_mode


config_optimizers = yaml.load(open(os.path.join("Control_Toolkit_ASF",
                                                "../../../Control_Toolkit_ASF/config_optimizers.yml")), Loader=yaml.FullLoader)
config_cost_function = yaml.load(open(os.path.join("Control_Toolkit_ASF",
                                                   "../../../Control_Toolkit_ASF/config_cost_function.yml")), Loader=yaml.FullLoader)
logger = get_logger(__name__)


class controller_fpga(template_controller):
    _has_optimizer = True
    
    def configure(self):
        raise NotImplementedError('Controller from FPGA not yet implemented')
        pass
        # TODO Establish connection with FPGA

    def step(self, s: np.ndarray, time=None, updated_attributes: "dict[str, TensorType]" = {}):
        self.update_attributes(updated_attributes)
        u = None
        #  TODO
        #   Send data (current state s, possibly also t) to FPGA
        #   Wait to receive control input u
        return u

    def controller_reset(self):
        pass
        
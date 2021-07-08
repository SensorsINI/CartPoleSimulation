
import scipy
import numpy as np
import pandas as pd
from Modeling.Pytorch.utilis_rnn import *

from Controllers.template_controller import template_controller
from CartPole.state_utilities import create_cartpole_state, cartpole_state_varname_to_index

import yaml, os
config = yaml.load(open(os.path.join('SI_Toolkit_ApplicationSpecificFiles', 'config.yml'), 'r'), Loader=yaml.FullLoader)

RNN_FULL_NAME = 'GRU-6IN-64H1-64H2-5OUT-0'
INPUTS_LIST = config['modeling']['PyTorch']['INPUTS_LIST']
OUTPUTS_LIST = config['modeling']['PyTorch']['OUTPUTS_LIST']
PATH_SAVE = config['modeling']['PyTorch']['PATH_SAVE']


class controller_rnn_as_mpc_pytorch(template_controller):
    def __init__(self):

        self.rnn_full_name = RNN_FULL_NAME
        self.path_save = PATH_SAVE

        self.device = get_device()

        # Create rnn instance and update lists of input, outputs and its name (if pretraind net loaded)
        self.net, self.rnn_name, self.inputs_list, self.outputs_list \
            = create_rnn_instance(load_rnn=self.rnn_full_name, path_save=self.path_save, device=self.device)

        self.normalization_info = load_normalization_info(self.path_save, self.rnn_full_name)

        self.net.reset()
        self.net.eval()

        self.rnn_input = pd.DataFrame(columns=self.inputs_list)
        self.rnn_output = pd.DataFrame(columns=self.outputs_list)


    def step(self, s, target_position, time=None):

        # Copy state and target_position into rnn_input

        if 'position' in self.rnn_input:
            self.rnn_input['position'] = [s[cartpole_state_varname_to_index('position')]]
        if 'angle' in self.rnn_input:
            self.rnn_input['angle'] = [s[cartpole_state_varname_to_index('angle')]]
        if 'positionD' in self.rnn_input:
            self.rnn_input['positionD'] = [s[cartpole_state_varname_to_index('positionD')]]
        if 'angleD' in self.rnn_input:
            self.rnn_input['angleD'] = [s[cartpole_state_varname_to_index('angleD')]]
        if 'target_position' in self.rnn_input:
            self.rnn_input['target_position'] = [target_position]

        rnn_input_normed = normalize_df(self.rnn_input, self.normalization_info)

        rnn_input_torch = torch.tensor(rnn_input_normed.values).float().unsqueeze(0).to(self.device)
        normalized_rnn_output = self.net(rnn_input=rnn_input_torch)
        normalized_rnn_output = normalized_rnn_output.detach().cpu().squeeze().tolist()
        normalized_rnn_output = pd.DataFrame(data=[normalized_rnn_output], columns=self.outputs_list)

        denormalized_rnn_output = denormalize_df(normalized_rnn_output, self.normalization_info)

        Q = float(denormalized_rnn_output['Q'])

        return Q

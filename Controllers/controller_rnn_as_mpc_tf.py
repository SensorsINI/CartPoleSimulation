from Modeling.TF.TF_Functions.Network import *

from CartPole._CartPole_mathematical_helpers import create_cartpole_state, cartpole_state_varname_to_index


RNN_FULL_NAME = 'GRU-5IN-64H1-64H2-1OUT-0'
INPUTS_LIST = ['s.position', 's.angle', 's.positionD', 's.angleD', 'target_position']
OUTPUTS_LIST = ['Q']
PATH_SAVE = './controllers/nets/rnn_as_mpc_tf/'

# TODO: For this moment it is just copied Pytorch version
class controller_rnn_as_mpc_tf:
    def __init__(self):

        # Create rnn instance and update lists of input, outputs and its name (if pretraind net loaded)
        self.net, self.rnn_name, self.inputs_list, self.outputs_list \
            = create_rnn_instance(load_rnn=RNN_FULL_NAME, path_save=PATH_SAVE,
                                  return_sequence=False, stateful=True,
                                  wash_out_len=1, batch_size=1)

        self.normalization_info = load_normalization_info(PATH_SAVE, RNN_FULL_NAME)

        self.rnn_input = pd.DataFrame(columns=self.inputs_list)
        self.rnn_output = pd.DataFrame(columns=self.outputs_list)


    def step(self, s, target_position, time=None):

        # Copy state and target_position into rnn_input

        if 's.position' in self.rnn_input:
            self.rnn_input['s.position'] = [s[cartpole_state_varname_to_index('position')]]
        if 's.angle' in self.rnn_input:
            self.rnn_input['s.angle'] = [s[cartpole_state_varname_to_index('angle')]]
        if 's.positionD' in self.rnn_input:
            self.rnn_input['s.positionD'] = [s[cartpole_state_varname_to_index('positionD')]]
        if 's.angleD' in self.rnn_input:
            self.rnn_input['s.angleD'] = [s[cartpole_state_varname_to_index('angleD')]]
        if 'target_position' in self.rnn_input:
            self.rnn_input['target_position'] = [target_position]

        rnn_input_normed = normalize_df(self.rnn_input, self.normalization_info)
        rnn_input_normed = np.squeeze(rnn_input_normed.to_numpy())
        rnn_input_normed = rnn_input_normed[np.newaxis, np.newaxis, :]
        normalized_rnn_output = self.net.predict_on_batch(rnn_input_normed)
        normalized_rnn_output = np.squeeze(normalized_rnn_output).tolist()
        normalized_rnn_output = copy.deepcopy(pd.DataFrame(data=[normalized_rnn_output], columns=self.outputs_list))
        denormalized_rnn_output = denormalize_df(normalized_rnn_output, self.normalization_info)

        Q = float(denormalized_rnn_output['Q'])

        return Q

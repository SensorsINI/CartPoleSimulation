import tensorflow as tf
from tensorflow import keras
from Modeling.SI_Toolkit.TF.TF_Functions.Network import create_rnn_instance

RNN_FULL_NAME = 'GRU-6IN-64H1-64H2-5OUT-0'
RNN_PATH = './save_tf/'
SAVEPATH = RNN_PATH + RNN_FULL_NAME + '/1/'

# Create rnn instance and update lists of input, outputs and its name (if pretraind net loaded)
net, rnn_name, rnn_inputs_names, rnn_outputs_names, normalization_info \
    = create_rnn_instance(load_rnn=RNN_FULL_NAME, path_save=RNN_PATH,
                          return_sequence=False, stateful=True,
                          wash_out_len=1, batch_size=1)

net_for_save = tf.keras.models.clone_model(net)

# tf.saved_model.save(net, SAVEPATH)
# loaded = tf.saved_model.load(SAVEPATH)
# print(list(loaded.signatures.keys()))  # ["serving_default"]
# infer = loaded.signatures["serving_default"]
# print(infer.structured_outputs)


net_for_save.save(SAVEPATH)
net_loaded = keras.models.load_model(SAVEPATH)
net_loaded.reset_states()
pass


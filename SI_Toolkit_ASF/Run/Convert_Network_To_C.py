# Convert_Network_To_C.py

""" Dense, GRU and LSTM networks with tanh activation on all but last layer are supported! """
from SI_Toolkit.C_implementation.TF2C import tf2C

path_to_models = '../Experiments/CP_Models_02_06_2025/Long/quant/'
net_name = 'LSTM-7IN-64H1-64H2-1OUT-0'
batch_size = 1

if __name__ == '__main__':
    tf2C(path_to_models=path_to_models, net_name=net_name, batch_size=batch_size)
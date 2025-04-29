# Convert_Network_To_C.py
""" Only dense and gru networks with tanh activation on all but last layer are supported! """
from SI_Toolkit.C_implementation.TF2C import tf2C

path_to_models = '../Experiments/Trial_14__17_08_2024/Models/'
net_name = 'Dense-7IN-32H1-32H2-1OUT-0'
batch_size = 1

if __name__ == '__main__':
    tf2C(path_to_models=path_to_models, net_name=net_name, batch_size=batch_size)
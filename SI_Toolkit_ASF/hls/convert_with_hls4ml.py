from types import SimpleNamespace

from SI_Toolkit.Functions.General.Initialization import get_net

import os
os.environ['PATH'] = '/tools/Xilinx/Vivado/2020.1/bin:' + os.environ['PATH']

# Parameters:
PATH_TO_MODELS = './SI_Toolkit_ASF/Experiments/NeuralImitatorCPS_v2/Models/'
NET_NAME = 'Dense-6IN-128H1-128H2-1OUT-1'

# Import network

a = SimpleNamespace()
batch_size = 1

a.path_to_models = PATH_TO_MODELS
a.net_name = NET_NAME

# Create a copy of the network suitable for inference (stateful and with sequence length one)
model, net_info = \
    get_net(a, time_series_length=1,
            batch_size=batch_size, stateful=True)

import hls4ml
config = hls4ml.utils.config_from_keras_model(model, granularity='model')
print("-----------------------------------")
hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                       hls_config=config,
                                                       output_dir='../../m1/prj0', ## !!!! If the path is longer it crashes. Depending on how long is the path it crashes at different places.
                                                       board='zybo-z7-20')  # TODO: What would be counterpart for Zybo

hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=None)

# TODO: Here could come a testing part...


# Synthesis
hls_model.build(csim=False)


# Reports
hls4ml.report.read_vivado_report('../m1/prj0/')
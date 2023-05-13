import yaml
import os
import hls4ml

config_hls = yaml.load(open(os.path.join('SI_Toolkit_ASF', 'config_hls.yml'), 'r'), Loader=yaml.FullLoader)

def convert_model_with_hls4ml(model, granularity='model'):
    config = hls4ml.utils.config_from_keras_model(model, granularity=granularity)


    config['Model']['Precision'] = config_hls['precision']
    config['Model']['Strategy'] = config_hls['Strategy']
    config['Model']['ReuseFactor'] = config_hls['ReuseFactor']

    hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                           hls_config=config,
                                                           output_dir=config_hls['output_dir'],
                                                           backend=config_hls['backend'],
                                                           ## !!!! If the path is long it crashes. Depending on how long is the path it crashes at different places.
                                                           part=config_hls['part'],                                                     # board=config_hls['board'],
    )

    hls_model.compile()

    return hls_model, config


# TODO: Not used yet as I don't have a way to feed dataset
#   Also the issue of setting precision for different granularity is not solved.
def hls4ml_numerical_model_profiling(model, data):

    hls_model, hls_model_config = convert_model_with_hls4ml(model, granularity='name')

    for layer in hls_model_config['LayerName'].keys():
        hls_model_config['LayerName'][layer]['Trace'] = True

    hls4ml.model.profiling.numerical(model=model, hls_model=hls_model, X=data)


import numpy as np

from hls4ml.converters.keras_to_hls import parse_default_keras_layer
from hls4ml.converters.keras_to_hls import keras_handler

from hls4ml.model.types import Quantizer
from hls4ml.model.types import IntegerPrecisionType


@keras_handler('Bidirectional')
def parse_bidirectional_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert(keras_layer["class_name"] == 'Bidirectional')

    layer = parse_default_keras_layer(keras_layer, input_names)

    layer['return_state'] = keras_layer['config']['layer']['config']['return_state'] #get layer['layer'] from bidirectional
    #layer['backward_layer'] = keras_layer['config']['backward_layer']
    layer['merge_mode'] = keras_layer['config']['merge_mode']

    layer['n_out_forward'] = keras_layer['config']['layer']['config']['units']
    layer['n_out_backward'] = keras_layer['config']['backward_layer']['config']['units']

    # forward_layer_unit = layer['layer']['config']['units']
    # backward_layer_unit = layer['backward_layer']['config']['units']

    #if layer['layer']['config']['return_state']:
    if layer['return_state']:
        # output_shape = [input_shapes[0][0], forward_layer_unit + backward_layer_unit]
        output_shape = [input_shapes[0][0], layer['n_out_forward'] + layer['n_out_backward']]
        # output_shape_1 = [input_shapes[0][0], forward_layer_unit]
        # output_shape_2 = [input_shapes[0][0], backward_layer_unit]
        # output_shape = [output_shape_0, output_shape_1, output_shape_2]
    else:
        # output_shape = [input_shapes[0][0], forward_layer_unit + backward_layer_unit]
        output_shape = [input_shapes[0][0], layer['n_out_forward'] + layer['n_out_backward']]
    return layer, output_shape
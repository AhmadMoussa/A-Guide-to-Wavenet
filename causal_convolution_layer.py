import tensorflow as tf
from create_variable import create_variable

def _create_causal_layer(input_batch):
        initial_filter_width = 32
        initial_channels = 2**8
        residual_channels = 16
        weights_filter = create_variable('filter', [initial_filter_width, initial_channels, residual_channels])
        return causal_conv(input_batch, weights_filter, 1)

# what this essentially does is reduce the number of channels and
def causal_conv(value, filter_, dilation, name='causal_conv'):
    filter_width = tf.shape(filter_)[0]
    restored = tf.nn.conv1d(value, filter_, stride=1, padding='VALID')
    tf.global_variables_initializer().run()
    # Remove excess elements at the end.
    out_width = tf.shape(value)[1] - (filter_width - 1) * dilation
    result = tf.slice(restored, [0, 0, 0], [-1, out_width, -1])
    return result

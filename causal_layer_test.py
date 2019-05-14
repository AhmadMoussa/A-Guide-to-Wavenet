import tensorflow as tf
from audio_loader_and_encoder import lqe

def causal_conv(value, filter_, dilation, name='causal_conv'):
    filter_width = tf.shape(filter_)[0]
    tf.initialize_all_variables().run()
    restored = tf.nn.conv1d(value, filter_, stride=1, padding='VALID')

    # Remove excess elements at the end.
    out_width = tf.shape(value)[1] - (filter_width - 1) * dilation
    result = tf.slice(restored, [0, 0, 0], [-1, out_width, -1])
    print("Everything ran")
    return result

def create_variable(name, shape):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable

def _create_causal_layer(input_batch):
        initial_filter_width = 32
        initial_channels = 2**8
        residual_channels = 16
        weights_filter = create_variable('filter', [initial_filter_width, initial_channels, residual_channels])
        return causal_conv(input_batch, weights_filter, 1)

# input here is a one hot encoded
def _create_network(input_batch):
    return _create_causal_layer(input_batch)


sess = tf.InteractiveSession()
loader = lqe()

one_hot_data = loader.get_oh().eval(session=sess)

print(_create_network(one_hot_data).eval(session = sess))

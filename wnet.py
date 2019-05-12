import tensorflow as tf

def create_variable(name, shape):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape = shape), name = name)
    return variable

class Wavenet():
    def __init__(dilations):
        self.batch_size = 1
        self.quantization_channels = 2**8
        self.filter_width = 2
        self.dilations = dilations
        self.residual_channels = 16
        self.dilation_channels = 32
        self.skip_channels = 16

        self.receptive_field = receptive_field_width(self.filter_width, self.dilations)
        self.variables = self._create_variables()

    @staticmethod
    def receptive_field_width(filter_width, dilations):
        return (self.filter_width - 1) * sum(self.dilations) + 1 + (filter_width - 1)


    # Let's create all the tensorflow variables used by the network
    def _create_variables(self):
        var = dict()
        with tf.variable_scope('wavenet'):
            with tf.variable_scope('causal_layer'):
                layer = dict()
                initial_channels = self.quantization_channels
                initial_filter_width = self.filter_width
                layer['filter'] = create_variable('filter',[initial_filter_width, initial_channels, self.residual_channels])
                var['causal_layer'] = layer

            var['dilated_stack'] = list()
            with tf.variable_scope('dilated_stack'):
                for i, dilation in enumerate(self.dilations):
                    with tf.variable_scope('layer{}'.format(i)):
                        current = dict()
                        current['filter'] = create_variable('filter', [self.filter_width, self.residual_channels, self.dilation_channels])
                        current['gate'] = create_variable('gate', [self.filter_width, self.residual_channels, self.dilation_channels])
                        current['dense'] = create_variable('dense', [1, self.dilation_channels, self.residual_channels])
                        current['skip'] = create_variable('skip', [1, self.dilation_channels, self.skip_channels])
                        var['dilated_stack'].append(current)

            with tf.variable_scope('postprocessing'):
                current = dict()
                current['postprocess1'] = create_variable('postprocess1', [1, self.skip_channels, self.skip_channels])
                current['postprocess2'] = create_variable('postprocess2', [1, self.skip_channels, self.quantization_channels])
                var['postprocessing'] = current

        return var

    def _create_causal_layer(self, input_batch):
        with tf.name_scope('causal_layer'):
            weights_filter = self.variables['causal_layer']['filter']
            return causal_conv(input_batch, weights_filter, 1)

    def _create_dilation_layer(self, input_batch, layer_index, dilation, output_width):
        variables = self.variables['dilated_stack'][layer_index]

        weights_filter = variables['filter']
        weights_gate = variables['gate']

        conv_filter = causal_conv(input_batch, weights_filter, dilation)
        conv_gate = causal_conv(input_batch, weights_gate, dilation)

        out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate) # Gated activation unit

        weights_dense = variables['dense'] # 1x1 convolution for residual output
        transformed = tf.nn.conv1d(out, weights_dense, stride = 1, padding = "SAME", name = "dense")

        skip_cut = tf.shape(out)[1] - output_width
        out_skip = tf.slice(out, [0, skip_cut, 0], [-1, 1, -1])
        weights_skip = variables['skip']
        skip_contribution = tf.nn.conv1d(out_skip, weights_skip, stride = 1, padding = "SAME", name = "skip")

        input_cut = tf.shape(input_batch)[1] - tf.shape(transformed)[1]
        input_batch = tf.slice(input_batch, [0, input_cut, 0], [-1, 1, -1])

        return skip_contribution, input_batch + transformed

    def _generator_conv(self, input_batch, state_batch, weights):
        past_weights = weights[0, :, :]
        curr_weights = weights[1, :, :]
        return tf.matmul(state_batch, past_weights) + tf.matmul(input_batch, curr_weights)

    def _generator_causal_layer(self, input_batch, state_batch):
        with tf.name_scope('causal_layer'):
            weights_filter = self.variables['causal_layer']['filter']
        return self._generator_conv(input_batch, state_batch, weights_filter)

    def _generator_dilation_layer(self, input_batch, state_batch, layer_index, dilation):
        variables = self.variables['dilated_stack']['layer_index']
        weights_filter = variables['filter']
        weights_gate = variables['gate']
        output_filter = self._generator_conv(input_batch, state_batch, weights_filter)
        output_gate = self._generator_conv(input_batch, state_batch, weights_gate)

        out = tf.tanh(output_filter) * tf.sigmoid(output_gate)

        weights_dense = variables['dense']
        transformed = tf.matmul(out, weights_dense[0, :, :])

        weights_skip = variables['skip']
        skip_contribution = tf.matmul(out, weights_skip[0, :, :])

        return skip_contribution, input_batch + transformed

    def _create_network(self, input_batch, global_condition_batch):
        outputs = []
        current_layer = input_batch

        current_layer = self._create_causal_layer(current_layer)
        output_width = tf.shape(input_batch)[1] - self.receptive_field + 1

        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index)):
                    output, current_layer = self._create_dilation_layer(current_layer, layer_index, dilation, output_width)
                    outputs.append(output)

        with tf.name_scope('postprocessing'):
            w1 = self.variables['postprocessing']['postprocess1']
            w2 = self.variables['postprocessing']['postprocess2']

            total = sum(outputs)
            transformed1 = tf.nn.relu(total)
            conv1 = tf.nn.conv1d(transformed1, w1, stride = 1, padding = "SAME")

            transformed2 = tf.nn.relu(conv1)
            conv2 = tf.nn.conv1d(transformed2, w2, stride = 1, padding = "SAME")

        return conv2

        def _create_generator(self, input_batch, global_condition_batch):
            init_ops = []
            push_ops = []
            outputs = []
            current_layer = input_batch

            q = tf.FIFOQueue(1, dtypes = tf.float32, shapes = (self.batch_size, self.quantization_channels))
            init = q.enqueue_many(tf.zeros((1, self.batch_size, self.qunatization_channels)))

            current_state = q.dequeue()
            push = q.enqueue([current_layer])
            init_ops.append(init)
            push_ops.append(push)

            current_layer = self._generator_causal_layer(current_layer, current_state)

            with tf.name_scope('dilated_stack'):
                for layer_index, dilation in enumerate(self.dilations):
                    with tf.name_scope('layer{}'.format(layer_index)):
                        q = tf.FIFOQueue(dilations, dtypes = tf.float32, shapes = (self.batch_size, self.residual_channels))
                        init = q.enqueue_many(tf.zeros((dilation, self.batch_size, self.residual_channels)))
                        current_state = q.dequeue()
                        push = q.enqueue([current_layer])
                        init_ops.append(init)
                        push_ops.append(push)

                        output, current_layer = self._generator_dilation_layer(current_layer, current_state, layer_index, dilation)
                        outputs.append(output)
            self.init_ops = init_ops
            self.push_ops = push_ops

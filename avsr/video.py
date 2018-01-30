import tensorflow as tf


def batch_norm_relu(inputs, is_training, data_format):
  inputs = tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      epsilon=1e-5, momentum=0.98,
      center=True, scale=True, training=is_training, fused=True)
  inputs = tf.nn.relu(inputs)
  return inputs


def fc_as_conv(inputs, kernel, filters):
    return tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel,
            padding='valid',
            use_bias=False,
            #activation=tf.nn.relu,
            kernel_initializer=tf.variance_scaling_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001),
        )


def conv2d_wrapper(inputs, filters, kernel_size, strides, data_format):
    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(scale=2),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001),
        data_format=data_format,
    )


def projection_shortcut(inputs, filters, strides, data_format):
    return conv2d_wrapper(inputs, filters, (1, 1), strides, data_format)


def residual_block(inputs, filters, kernel_size, strides, data_format, is_training, project_shortcut=False):

    shortcut = inputs

    inputs = batch_norm_relu(inputs, is_training, data_format)

    if project_shortcut is True:
        shortcut = projection_shortcut(inputs, filters, strides, data_format)

    inputs = conv2d_wrapper(inputs, filters, kernel_size, strides, data_format)
    inputs = batch_norm_relu(inputs, is_training, data_format)
    inputs = conv2d_wrapper(inputs, filters, kernel_size, 1, data_format)

    return inputs + shortcut


def my_2d_cnn():

    def model(inputs, is_training, cnn_dense_units, cnn_filters):
        data_format = 'channels_last'
        # inputs = tf.transpose(inputs, [0, 3, 1, 2])
        flow = (inputs * 2) - 1

        ### debug images
        # dbg_img = tf.summary.image(
        #     name="input_images",
        #     tensor=layer_input,
        #     max_outputs=1000
        # )
        ###

        for layer_id, num_filters in enumerate(cnn_filters):

            flow = conv2d_wrapper(
                inputs=flow,
                filters=num_filters,
                kernel_size=(3, 3),
                strides=1 if layer_id == 0 else 2,
                data_format=data_format
            )

            flow = batch_norm_relu(flow, is_training, data_format)

        # conv_flat = tf.contrib.layers.flatten(layer_output)

        final = fc_as_conv(flow, [9, 9], cnn_dense_units)
        final = tf.squeeze(final, axis=[1,2])

        return final

    return model


def my_resnet_cnn():

    def model(inputs, is_training, cnn_dense_units, cnn_filters):
        data_format = 'channels_last'

        flow = (inputs * 2) - 1

        flow = conv2d_wrapper(flow, cnn_filters[0], (3, 3), 1, data_format)

        flow = residual_block(flow, cnn_filters[0], (3, 3), 1, data_format, is_training, project_shortcut=False)

        for layer_id, num_filters in enumerate(cnn_filters[1:]):
            flow = residual_block(flow, num_filters, (3, 3), 2, data_format, is_training, project_shortcut=True)

        final = fc_as_conv(flow, flow.get_shape().as_list()[1:-1], cnn_dense_units)
        final = tf.squeeze(final, axis=[1, 2])

        return final

    return model


def my_3d_cnn():

    def model(inputs, is_training, cnn_dense_units, cnn_filters):
        data_format = 'channels_last'
        # inputs = tf.transpose(inputs, [0, 4, 1, 2, 3])
        conv = inputs

        for layer, num_filters in enumerate(cnn_filters):
            conv = tf.layers.conv3d(
                conv,
                filters=num_filters,
                kernel_size=[5, 4, 4] if layer==0 else (1,3,3),
                strides=(1,1,1) if layer==0 else (1, 2, 2),
                padding='valid',
                activation=tf.nn.relu,
                use_bias=True,
                bias_initializer=tf.variance_scaling_initializer(),
                kernel_initializer=tf.variance_scaling_initializer(),
                # kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                data_format=data_format,
            )
            conv = tf.layers.dropout(conv, rate=0.1, training=is_training)

        bs, ts, _, _, _ = tf.unstack(tf.shape(conv))
        _, _, w, h, f = conv.get_shape()
        conv4_flat = tf.reshape(conv, [bs*ts, w*h*f])
        final = tf.layers.dense(conv4_flat, cnn_dense_units, activation=None)

        final = tf.reshape(final, [bs, ts, cnn_dense_units])

        return final

    return model


def cnn_layers(inputs, cnn_type, is_training, cnn_filters, cnn_dense_units=128):

    # inputs = tf.expand_dims(inputs, axis=-1)
    bs, ts, _, _, _ = tf.unstack(tf.shape(inputs))
    _, _, height, width, chans = inputs.get_shape().as_list()

    if cnn_type == 'resnet_cnn':
        inputs = tf.reshape(inputs, shape=[-1, int(height), int(width), int(chans)])
        model = my_resnet_cnn()
        outputs = model(inputs, is_training=is_training, cnn_dense_units=cnn_dense_units, cnn_filters=cnn_filters)

        outputs = tf.reshape(outputs, [bs, ts, cnn_dense_units])  # unwrap
    elif cnn_type == '2dconv':

        with tf.device('/gpu:0'):
            inputs = tf.reshape(inputs, shape=[bs * ts, int(height), int(width), int(chans)])  # wrap
            model = my_2d_cnn()
            outputs = model(inputs, is_training=is_training, cnn_filters=cnn_filters, cnn_dense_units=cnn_dense_units)
            outputs = tf.reshape(outputs, [bs, ts, cnn_dense_units])  # unwrap

    elif cnn_type == '3dconv':
        with tf.device('/gpu:0'):
            model = my_3d_cnn()
            outputs = model(inputs, is_training=is_training, cnn_dense_units=cnn_dense_units, cnn_filters=cnn_filters)
    else:
        raise Exception('undefined CNN, did you mean `resnet` ?')


    return outputs

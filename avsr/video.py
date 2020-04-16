import tensorflow as tf


def batch_norm_relu(inputs, is_training, data_format, name=None):
    r"""Performs a batch normalization followed by a ReLU."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    inputs = tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else -1,
        epsilon=1e-5, momentum=0.98,
        center=True, scale=True, training=is_training, fused=True,
        name=name+'_bn' if name is not None else name)
    inputs = tf.nn.relu(inputs, name=name+'_relu' if name is not None else name)
    return inputs


def conv2d_wrapper(inputs, filters, kernel_size, strides, data_format, padding='SAME', activation=None, name=None):
    out =  tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=True,
        kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_in'),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.001),
        data_format=data_format,
        activation=activation,
        name=name
    )
    return out


def conv3d_wrapper(inputs, filters, kernel_size, strides, data_format, padding='SAME', activation=None):
    return tf.layers.conv3d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_in'),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.001),
        activation=activation
    )


def projection_shortcut(inputs, filters, strides, data_format, padding='SAME', name=None):
    return conv2d_wrapper(inputs, filters, (1, 1), strides, data_format, padding=padding, name=name)


def projection_shortcut_3d(inputs, filters, strides, data_format, padding='SAME'):
    return conv3d_wrapper(inputs, filters, (1, 1, 1), strides, data_format, padding=padding)


def residual_block(inputs, filters, kernel_size, strides, data_format, is_training,
                   project_shortcut=False, skip_bn=False, name=None):
    r"""
    Each residual block contains two convolutions, performing the function:
    output = projection(input) + conv2(conv1(inputs))
    Parameters
    ----------
    inputs
    filters
    kernel_size
    strides
    data_format
    is_training
    project_shortcut
    skip_bn

    Returns
    -------

    """
    shortcut = inputs

    if skip_bn is False:
        inputs = batch_norm_relu(inputs, is_training, data_format, name=name+'_first')

    if project_shortcut is True:
        shortcut = projection_shortcut(shortcut, filters, strides, data_format, name=name+'_shortcut')

    inputs = conv2d_wrapper(inputs, filters, kernel_size, strides, data_format, name=name+'_conv1')
    inputs = batch_norm_relu(inputs, is_training, data_format, name=name+'_second')
    inputs = conv2d_wrapper(inputs, filters, kernel_size, (1, 1), data_format, name=name+'_conv2')

    return tf.add(inputs, shortcut, name=name)


def residual_block_3d(inputs, filters, kernel_size, strides, data_format, is_training, project_shortcut=False, skip_bn=False):
    shortcut = inputs

    if skip_bn is False:
        inputs = batch_norm_relu(inputs, is_training, data_format)

    if project_shortcut is True:
        shortcut = projection_shortcut_3d(shortcut, filters, strides, data_format)

    inputs = conv3d_wrapper(inputs, filters, kernel_size, strides, data_format)
    inputs = batch_norm_relu(inputs, is_training, data_format)
    inputs = conv3d_wrapper(inputs, filters, kernel_size, (1, 1, 1), data_format)

    return inputs + shortcut


def conv2d_cnn():

    def model(inputs, is_training, cnn_dense_units, cnn_filters):
        data_format = 'channels_first'
        # inputs = tf.transpose(inputs, [0, 3, 1, 2])
        flow = (inputs * 2) - 1

        # debug images
        # dbg_img = tf.summary.image(
        #     name="input_images",
        #     tensor=layer_input,
        #     max_outputs=1000
        # )
        #

        for layer_id, num_filters in enumerate(cnn_filters):

            flow = conv2d_wrapper(
                inputs=flow,
                filters=num_filters,
                kernel_size=(3, 3),
                strides=(1, 1) if layer_id == 0 else (2, 2),
                data_format=data_format
            )

            flow = batch_norm_relu(flow, is_training, data_format)

        final = conv2d_wrapper(flow, cnn_dense_units, flow.get_shape().as_list()[1:-1], (1, 1), data_format, 'valid', tf.nn.relu)
        final = tf.squeeze(final, axis=[1, 2])

        return final

    return model


def resnet_cnn():

    def model(inputs, is_training, cnn_dense_units, cnn_filters):
        data_format = 'channels_last'

        # random_flip = lambda img: tf.image.random_flip_left_right(img, seed=1001)
        # random_contrast = lambda img: tf.image.random_contrast(img, lower=0.8, upper=1.2, seed=1002)
        # random_brightness = lambda img: tf.image.random_brightness(img, max_delta=0.2, seed=1003)
        # random_hue = lambda img: tf.image.random_hue(img, max_delta=0.2, seed=1004)
        # random_sat = lambda img: tf.image.random_saturation(img, lower=0.8, upper=1.2, seed=1005)
        #
        # if is_training is True:
        #     inputs = tf.map_fn(lambda img:
        #                        random_flip(img),
        #                            random_sat(
        #                                random_hue(
        #                                    random_brightness(
        #                                        random_contrast(img))))),
        #                        inputs, back_prop=False, parallel_iterations=64)

        # flow = (inputs * 2) - 1  # the record file should contain already normalised pixel values
        if data_format == 'channels_first':
            flow = tf.transpose(inputs, [0, 3, 1, 2])
        else:
            flow = inputs
        flow = conv2d_wrapper(flow, cnn_filters[0], (3, 3), (1, 1), data_format=data_format, name='layer0')
        flow = batch_norm_relu(flow, is_training, data_format, name='layer0')

        flow = tf.identity(flow, name='identity')  # ??
        flow = residual_block(flow, cnn_filters[0], (3, 3), (1, 1), data_format,
                              is_training, project_shortcut=False, skip_bn=True,
                              name='res_block_0')

        for layer_id, num_filters in enumerate(cnn_filters[1:]):
            flow = residual_block(flow, num_filters, (3, 3), (2, 2), data_format, is_training, project_shortcut=True,
                                  name='res_block_'+str(layer_id+1))
            # flow = residual_block(flow, num_filters, (3, 3), 1, data_format, is_training, project_shortcut=False)
            # flow = residual_block(flow, num_filters, (3, 3), 1, data_format, is_training, project_shortcut=False)

        if data_format == 'channels_first':
            kernel = flow.get_shape().as_list()[2:4]
            squeeze_axis = [2, 3]
        else:  # channels_last
            kernel = flow.get_shape().as_list()[1:-1]
            squeeze_axis = [1, 2]

        final = conv2d_wrapper(flow, cnn_dense_units, kernel, (1,1), data_format, 'VALID', tf.nn.relu, name='flatten')
        final = tf.squeeze(final, axis=squeeze_axis)

        final = tf.identity(final, name='identity2')  # ??
        return final

    return model


def conv3d_cnn():

    def model(inputs, is_training, cnn_dense_units, cnn_filters):
        data_format = 'channels_last'
        # inputs = tf.transpose(inputs, [0, 4, 1, 2, 3])
        flow = (inputs * 2) - 1

        flow = conv3d_wrapper(flow, cnn_filters[0], (1, 3, 3), (1, 1, 1), data_format)
        flow = batch_norm_relu(flow, is_training, data_format)

        flow = residual_block_3d(flow, cnn_filters[0], (3, 3, 3), 1, data_format, is_training, project_shortcut=False,
                                 skip_bn=True)

        for layer_id, num_filters in enumerate(cnn_filters[1:]):
            flow = residual_block_3d(flow, num_filters, (3, 3, 3), (1, 2, 2), data_format, is_training, project_shortcut=True)
            # increase depth here
            # flow = residual_block(flow, num_filters, (3, 3), 1, data_format, is_training, project_shortcut=False)

        final = conv3d_wrapper(flow, cnn_dense_units, [1] + flow.get_shape().as_list()[2:-1], (1, 1), data_format, 'VALID', tf.nn.relu)
        final = tf.squeeze(final, axis=[2, 3])

        return final

    return model


def cnn_layers(inputs, cnn_type, is_training, cnn_filters, cnn_dense_units=128):

    bs, ts, _, _, _ = tf.unstack(tf.shape(inputs))
    _, _, height, width, chans = inputs.get_shape().as_list()

    if cnn_type == 'resnet_cnn':
        inputs = tf.reshape(inputs, shape=[-1, int(height), int(width), int(chans)])
        model = resnet_cnn()
        outputs = model(inputs, is_training=is_training, cnn_dense_units=cnn_dense_units, cnn_filters=cnn_filters)
        outputs = tf.reshape(outputs, [bs, ts, cnn_dense_units])  # unwrap

    elif cnn_type == '2dconv_cnn':
        inputs = tf.reshape(inputs, shape=[bs * ts, int(height), int(width), int(chans)])  # wrap
        model = conv2d_cnn()
        outputs = model(inputs, is_training=is_training, cnn_filters=cnn_filters, cnn_dense_units=cnn_dense_units)
        outputs = tf.reshape(outputs, [bs, ts, cnn_dense_units])  # unwrap

    elif cnn_type == '3dconv_cnn':
        model = conv3d_cnn()
        outputs = model(inputs, is_training=is_training, cnn_dense_units=cnn_dense_units, cnn_filters=cnn_filters)

    else:
        raise Exception('undefined CNN, did you mean `resnet_cnn` ?')

    return outputs

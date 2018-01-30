from tensorflow.contrib.rnn import MultiRNNCell,DeviceWrapper, DropoutWrapper, \
    LSTMCell, GRUCell, LSTMBlockCell, UGRNNCell
import tensorflow as tf


def _build_single_cell(cell_type, num_units, use_dropout, mode, dropout_probability, device=None):
    r"""

    :param num_units: `int`
    :return:
    """
    if cell_type == 'lstm':
        cells = LSTMCell(num_units=num_units,
                         use_peepholes=True,
                         cell_clip=10.0,
                         initializer=tf.variance_scaling_initializer(),

                         )
    elif cell_type == 'gru':
        cells = GRUCell(num_units=num_units,
                        kernel_initializer=tf.variance_scaling_initializer(),
                        bias_initializer=tf.variance_scaling_initializer())
    elif cell_type == 'ugrnn':
        cells = UGRNNCell(num_units)
    elif cell_type == 'lstm_block':
        cells = LSTMBlockCell(num_units=num_units,
                              use_peephole=True,
                              cell_clip=10.0,)
    else:
        raise Exception('cell type not supported: {}'.format(cell_type))

    if use_dropout is True and mode == 'train':
        cells = DropoutWrapper(cells,
                               input_keep_prob=dropout_probability[0],
                               state_keep_prob=dropout_probability[1],
                               output_keep_prob=dropout_probability[2],
                               variational_recurrent=False,
                               # dtype=tf.float32,
                               # input_size=self._inputs.get_shape()[1:],
                               )
    if device is not None:
        cells = DeviceWrapper(cells, device=device)

    return cells


def build_rnn_layers(
        cell_type,
        num_units_per_layer,
        use_dropout,
        dropout_probability,
        mode,
        base_gpu,
    ):
    if base_gpu:
        device = '/gpu:{}'.format(base_gpu)
    else:
        device = None

    cell_list = []
    for layer, units in enumerate(num_units_per_layer):

        cell = _build_single_cell(
            cell_type=cell_type,
            num_units=units,
            use_dropout=use_dropout,
            dropout_probability=dropout_probability,
            mode=mode,
            device=device)

        cell_list.append(cell)

    if len(cell_list) == 1:
        return cell_list[0]
    else:
        return MultiRNNCell(cell_list)
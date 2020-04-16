import tensorflow as tf
from tensorflow.contrib.rnn import MultiRNNCell, DeviceWrapper, DropoutWrapper, \
    LSTMCell, GRUCell, LSTMBlockCell, UGRNNCell, NASCell, GRUBlockCellV2, \
    HighwayWrapper, ResidualWrapper

from tensorflow.contrib.rnn.python.ops.rnn_cell import LayerNormLSTMCell, LayerNormBasicLSTMCell
def _build_single_cell(cell_type, num_units, use_dropout, mode, dropout_probability, dtype, device=None):
    r"""

    :param num_units: `int`
    :return:
    """
    if cell_type == 'lstm':
        cells = LSTMCell(num_units=num_units,
                         use_peepholes=False,
                         cell_clip=1.0,
                         initializer=tf.variance_scaling_initializer(),
                         dtype=dtype)
    elif cell_type == 'layernorm_lstm':
        cells = LayerNormLSTMCell(num_units=num_units,
                                  cell_clip=1.0)
    elif cell_type == 'layernorm_basiclstm':
        cells = LayerNormBasicLSTMCell(num_units=num_units)
    elif cell_type == 'gru':
        cells = GRUCell(num_units=num_units,
                        kernel_initializer=tf.variance_scaling_initializer(),
                        bias_initializer=tf.variance_scaling_initializer(),
                        dtype=dtype
                        )
    elif cell_type == 'ugrnn':
        cells = UGRNNCell(num_units)
    elif cell_type == 'lstm_block':
        cells = LSTMBlockCell(num_units=num_units,
                              use_peephole=True,
                              cell_clip=None)
    elif cell_type == 'gru_block':
        cells = GRUBlockCellV2(num_units=num_units)
    elif cell_type == 'nas':
        cells = NASCell(num_units=num_units)
    elif cell_type == 'lstm_masked':
        from tensorflow.contrib.model_pruning import MaskedLSTMCell
        cells = MaskedLSTMCell(num_units=num_units)
    else:
        raise Exception('cell type not supported: {}'.format(cell_type))

    if use_dropout is True and mode == 'train':
        cells = DropoutWrapper(cells,
                               input_keep_prob=dropout_probability[0],
                               state_keep_prob=dropout_probability[1],
                               output_keep_prob=dropout_probability[2],
                               variational_recurrent=False,
                               dtype=dtype,
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
        dtype,
        residual_connections=False,
        highway_connections=False,
        weight_sharing=False,
        as_list=False,
    ):

    cell_list = []
    for layer, units in enumerate(num_units_per_layer):

        if layer > 1 and weight_sharing is True:
            cell = cell_list[-1]
        else:
            cell = _build_single_cell(
                cell_type=cell_type,
                num_units=units,
                use_dropout=use_dropout,
                dropout_probability=dropout_probability,
                mode=mode,
                dtype=dtype,
            )

            if highway_connections is True and layer > 0:
                cell = HighwayWrapper(cell)
            elif residual_connections is True and layer > 0:
                cell = ResidualWrapper(cell)

        cell_list.append(cell)

    if len(cell_list) == 1:
        return cell_list[0]
    else:
        if as_list is False:
            return MultiRNNCell(cell_list)
        else:
            return cell_list

import tensorflow as tf
import collections
from .cells import build_rnn_layers, create_attention_mechanism
from tensorflow.contrib import seq2seq
from tensorflow.contrib.rnn import MultiRNNCell


from tensorflow.python.layers.core import Dense


class EncoderData(collections.namedtuple("EncoderData", ("outputs", "final_state"))):
    pass


class Seq2SeqEncoder(object):

    def __init__(self,
                 data,
                 mode,
                 hparams,
                 gpu_id):

        self._data = data
        self._mode = mode
        self._hparams = hparams
        self._gpu_id = gpu_id

        self._init_data()
        self._init_encoder()

    def _init_data(self):
        self._inputs = self._data.inputs
        self._inputs_len = self._data.inputs_length

        # self._labels = self._data.labels
        # self._labels_len = self._data.labels_length

        if self._hparams.batch_normalisation is True:
            self._inputs = tf.layers.batch_normalization(
                inputs=self._inputs,
                axis=-1,
                training=(self._mode == 'train'),
                fused=True,
            )

    def _init_encoder(self):
        r"""
        Instantiates the seq2seq encoder
        :return:
        """
        with tf.variable_scope("Encoder") as scope:

            encoder_inputs = self._maybe_add_dense_layers()
            # encoder_inputs = a_resnet(encoder_inputs, self._mode == 'train')

            if self._hparams.encoder_type == 'unidirectional':
                self._encoder_cells = build_rnn_layers(
                    cell_type=self._hparams.cell_type,
                    num_units_per_layer=self._hparams.encoder_units_per_layer,
                    use_dropout=self._hparams.use_dropout,
                    dropout_probability=self._hparams.dropout_probability,
                    mode=self._mode,
                    residual_connections=self._hparams.residual_encoder,
                    highway_connections=self._hparams.highway_encoder,
                    base_gpu=self._gpu_id,
                    dtype=self._hparams.dtype,
                )

                self._encoder_outputs, self._encoder_final_state = tf.nn.dynamic_rnn(
                    cell=self._encoder_cells,
                    inputs=encoder_inputs,
                    sequence_length=self._inputs_len,
                    parallel_iterations=self._hparams.batch_size[0 if self._mode == 'train' else 1],
                    swap_memory=False,
                    dtype=self._hparams.dtype,
                    scope=scope,
                    )

            elif self._hparams.encoder_type == 'bidirectional':

                self._fw_cells = build_rnn_layers(
                    cell_type=self._hparams.cell_type,
                    num_units_per_layer=self._hparams.encoder_units_per_layer,
                    use_dropout=self._hparams.use_dropout,
                    dropout_probability=self._hparams.dropout_probability,
                    mode=self._mode,
                    base_gpu=self._gpu_id,
                    dtype=self._hparams.dtype,
                )

                self._bw_cells = build_rnn_layers(
                    cell_type=self._hparams.cell_type,
                    num_units_per_layer=self._hparams.encoder_units_per_layer,
                    use_dropout=self._hparams.use_dropout,
                    dropout_probability=self._hparams.dropout_probability,
                    mode=self._mode,
                    base_gpu=self._gpu_id,
                    dtype=self._hparams.dtype,
                )

                bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=self._fw_cells,
                    cell_bw=self._bw_cells,
                    inputs=encoder_inputs,
                    sequence_length=self._inputs_len,
                    dtype=self._hparams.dtype,
                    parallel_iterations=self._hparams.batch_size[0 if self._mode == 'train' else 1],
                    swap_memory=False,
                    scope=scope,

                )

                self._encoder_outputs = tf.concat(bi_outputs, -1)
                encoder_state = []

                for layer in range(len(bi_state[0])):
                    fw_state = bi_state[0][layer]
                    bw_state = bi_state[1][layer]

                    if self._hparams.cell_type == 'gru':
                        cat = tf.concat([fw_state, bw_state], axis=-1)
                        proj = tf.layers.dense(cat, units=self._hparams.decoder_units_per_layer[0], use_bias=False)
                        encoder_state.append(proj)
                    elif self._hparams.cell_type == 'lstm':
                        cat_c = tf.concat([fw_state.c, bw_state.c], axis=-1)
                        cat_h = tf.concat([fw_state.h, bw_state.h], axis=-1)
                        proj_c = tf.layers.dense(cat_c, units=self._hparams.decoder_units_per_layer[0], use_bias=False)
                        proj_h = tf.layers.dense(cat_h, units=self._hparams.decoder_units_per_layer[0], use_bias=False)
                        state_tuple = tf.contrib.rnn.LSTMStateTuple(c=proj_c, h=proj_h)
                        encoder_state.append(state_tuple)
                    else:
                        raise ValueError('BiRNN fusion strategy not implemented for this cell')
                encoder_state = tuple(encoder_state)

                self._encoder_final_state = encoder_state

            else:
                raise Exception('Allowed encoder types: `unidirectional`, `bidirectional`')

    def _maybe_add_dense_layers(self):
        r"""
        Optionally passes self._input through several Fully Connected (Dense) layers
        with the configuration defined by the self._input_dense_layers tuple

        Returns
        -------
        The output of the network of Dense layers
        """
        layer_inputs = self._inputs
        if self._hparams.input_dense_layers[0] > 0:

            fc = [Dense(units,
                        activation=tf.nn.selu,
                        use_bias=False,
                        kernel_initializer=tf.variance_scaling_initializer(),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001))
                  for units in self._hparams.input_dense_layers]

            for layer in fc:
                layer_inputs = layer(layer_inputs)
        else:
            pass
        return layer_inputs

    def get_data(self):

        return EncoderData(
            outputs=self._encoder_outputs,
            final_state=self._encoder_final_state
        )


class AttentiveEncoder(Seq2SeqEncoder):

    def __init__(self,
                 data,
                 mode,
                 hparams,
                 gpu_id,
                 attended_memory,
                 attended_memory_length):
        r"""
        Implements https://arxiv.org/abs/1809.01728
        """

        self._attended_memory = attended_memory
        self._attended_memory_length = attended_memory_length

        super(AttentiveEncoder, self).__init__(
            data,
            mode,
            hparams,
            gpu_id)

    def _init_encoder(self):
        with tf.variable_scope("Encoder") as scope:

            encoder_inputs = self._maybe_add_dense_layers()

            if self._hparams.encoder_type == 'unidirectional':
                self._encoder_cells = build_rnn_layers(
                    cell_type=self._hparams.cell_type,
                    num_units_per_layer=self._hparams.encoder_units_per_layer,
                    use_dropout=self._hparams.use_dropout,
                    dropout_probability=self._hparams.dropout_probability,
                    mode=self._mode,
                    base_gpu=self._gpu_id,
                    as_list=True,
                    dtype=self._hparams.dtype)

                attention_mechanism, output_attention = create_attention_mechanism(
                    attention_type=self._hparams.attention_type[0][0],
                    num_units=self._hparams.encoder_units_per_layer[-1],
                    memory=self._attended_memory,
                    memory_sequence_length=self._attended_memory_length,
                    mode=self._mode,
                    dtype=self._hparams.dtype
                )

                attention_cells = seq2seq.AttentionWrapper(
                    cell=self._encoder_cells[-1],
                    attention_mechanism=attention_mechanism,
                    attention_layer_size=self._hparams.decoder_units_per_layer[-1],
                    alignment_history=False,
                    output_attention=output_attention,
                )

                self._encoder_cells[-1] = attention_cells

                self._encoder_outputs, self._encoder_final_state = tf.nn.dynamic_rnn(
                    cell=MultiRNNCell(self._encoder_cells),
                    inputs=encoder_inputs,
                    sequence_length=self._inputs_len,
                    parallel_iterations=self._hparams.batch_size[0 if self._mode == 'train' else 1],
                    swap_memory=False,
                    dtype=self._hparams.dtype,
                    scope=scope,
                    )

    def get_data(self):

        return EncoderData(
            outputs=self._encoder_outputs,
            final_state=(self._encoder_final_state[:-1], self._encoder_final_state[-1].cell_state)
        )

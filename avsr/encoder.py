import tensorflow as tf
import collections
from .cells import build_rnn_layers
from tensorflow.contrib.rnn import MultiRNNCell
from .attention import add_attention

from tensorflow.python.layers.core import Dense


class EncoderData(collections.namedtuple("EncoderData", ("outputs", "final_state"))):
    pass


class Seq2SeqEncoder(object):

    def __init__(self,
                 data,
                 mode,
                 hparams,
                 num_units_per_layer,
                 dropout_probability,
                 **kwargs
                 ):

        self._data = data
        self._mode = mode
        self._hparams = hparams
        self._num_units_per_layer = num_units_per_layer
        self._dropout_probability = dropout_probability

        self._init_data()
        self._init_encoder()

        if kwargs.get('regress_aus', False) and mode == 'train':
            self._init_au_loss()

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
        if self._hparams.instance_normalisation is True:
            from tensorflow.contrib.layers import instance_norm
            self._inputs = instance_norm(
                inputs=self._inputs,
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
                    num_units_per_layer=self._num_units_per_layer,
                    use_dropout=self._hparams.use_dropout,
                    dropout_probability=self._dropout_probability,
                    mode=self._mode,
                    residual_connections=self._hparams.residual_encoder,
                    highway_connections=self._hparams.highway_encoder,
                    dtype=self._hparams.dtype,
                    weight_sharing=self._hparams.encoder_weight_sharing,
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
                    num_units_per_layer=self._num_units_per_layer,
                    use_dropout=self._hparams.use_dropout,
                    dropout_probability=self._dropout_probability,
                    mode=self._mode,
                    dtype=self._hparams.dtype,
                )

                self._bw_cells = build_rnn_layers(
                    cell_type=self._hparams.cell_type,
                    num_units_per_layer=self._num_units_per_layer,
                    use_dropout=self._hparams.use_dropout,
                    dropout_probability=self._dropout_probability,
                    mode=self._mode,
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

    def _init_au_loss(self):
        encoder_output_layer = tf.layers.Dense(
            units=2, activation=tf.nn.sigmoid,
            )

        projected_outputs = encoder_output_layer(self._encoder_outputs)
        normed_aus = tf.clip_by_value(self._data.payload['aus'], 0.0, 3.0) / 3.0

        mask = tf.sequence_mask(self._inputs_len, dtype=self._hparams.dtype)
        mask = tf.expand_dims(mask, -1)
        mask = tf.tile(mask, [1, 1, 2])

        self.au_loss = tf.losses.mean_squared_error(
            predictions=projected_outputs,
            labels=normed_aus,
            weights=mask
        )

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
                 num_units_per_layer,
                 attended_memory,
                 attended_memory_length,
                 dropout_probability):
        r"""
        Implements https://arxiv.org/abs/1809.01728
        """

        self._attended_memory = attended_memory
        self._attended_memory_length = attended_memory_length

        super(AttentiveEncoder, self).__init__(
            data,
            mode,
            hparams,
            num_units_per_layer,
            dropout_probability
        )

    def _init_encoder(self):
        with tf.variable_scope("Encoder") as scope:

            encoder_inputs = self._maybe_add_dense_layers()

            if self._hparams.encoder_type == 'unidirectional':
                self._encoder_cells = build_rnn_layers(
                    cell_type=self._hparams.cell_type,
                    num_units_per_layer=self._num_units_per_layer,
                    use_dropout=self._hparams.use_dropout,
                    dropout_probability=self._dropout_probability,
                    mode=self._mode,
                    as_list=True,
                    dtype=self._hparams.dtype)

                self._encoder_cells = maybe_list(self._encoder_cells)

                #### here weird code

                # 1. reverse mem
                # self._attended_memory = tf.reverse(self._attended_memory, axis=[1])

                # 2. append zeros
                # randval1 = tf.random.uniform(shape=[], minval=25, maxval=100, dtype=tf.int32)
                # randval2 = tf.random.uniform(shape=[], minval=25, maxval=100, dtype=tf.int32)
                # zeros_slice1 = tf.zeros([1, randval1, 256], dtype=tf.float32)  # assuming we use inference on a batch size of 1
                # zeros_slice2 = tf.zeros([1, randval2, 256], dtype=tf.float32)
                # self._attended_memory = tf.concat([zeros_slice1, self._attended_memory, zeros_slice2], axis=1)
                # self._attended_memory_length += randval1 + randval2

                # 3. blank mem
                # self._attended_memory = 0* self._attended_memory

                # 4. mix with noise
                # noise = tf.random.truncated_normal(shape=tf.shape(self._attended_memory))
                # noise = tf.random.uniform(shape=tf.shape(self._attended_memory))

                # self._attended_memory = noise

                #### here stop weird code

                attention_cells, dummy_initial_state = add_attention(
                    cells=self._encoder_cells[-1],
                    attention_types=self._hparams.attention_type[0],
                    num_units=self._num_units_per_layer[-1],
                    memory=self._attended_memory,
                    memory_len=self._attended_memory_length,
                    mode=self._mode,
                    dtype=self._hparams.dtype,
                    batch_size=tf.shape(self._inputs_len),
                    write_attention_alignment=self._hparams.write_attention_alignment,
                    fusion_type='linear_fusion',
                )

                self._encoder_cells[-1] = attention_cells

                self._encoder_cells = maybe_multirnn(self._encoder_cells)

                self._encoder_outputs, self._encoder_final_state = tf.nn.dynamic_rnn(
                    cell=self._encoder_cells,
                    inputs=encoder_inputs,
                    sequence_length=self._inputs_len,
                    parallel_iterations=self._hparams.batch_size[0 if self._mode == 'train' else 1],
                    swap_memory=False,
                    dtype=self._hparams.dtype,
                    scope=scope,
                    )

                if self._hparams.write_attention_alignment is True:
                    # self.weights_summary = self._encoder_final_state[-1].attention_weight_history.stack()
                    self.attention_summary, self.attention_alignment = self._create_attention_alignments_summary(maybe_list(self._encoder_final_state)[-1])

    def _create_attention_alignments_summary(self, states):
        r"""
        Generates the alignment images, useful for visualisation/debugging purposes
        """
        attention_alignment = states.alignment_history[0].stack()

        attention_alignment = tf.expand_dims(tf.transpose(attention_alignment, [1, 2, 0]), -1)

        # attention_images_scaled = tf.image.resize_images(1-attention_images, (256,128))
        attention_images = 1 - attention_alignment

        attention_summary = tf.summary.image("attention_images_cm", attention_images,
                                             max_outputs=self._hparams.batch_size[1])

        return attention_summary, attention_alignment

    def get_data(self):

        def prepare_final_state(state):
            r"""
            state is a stack of zero or several RNN cells, followed by a final Attention wrapped RNN cell
            """

            from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import AttentionWrapperState

            final_state = []
            if type(state) is tuple:
                for cell in state:
                    if type(cell) == AttentionWrapperState:
                        final_state.append(cell.cell_state)
                    else:
                        final_state.append(cell)
                return final_state
            else:  # only one RNN layer of attention wrapped cells
                return state.cell_state

        return EncoderData(
            outputs=self._encoder_outputs,
            final_state=prepare_final_state(self._encoder_final_state)
        )


def maybe_list(obj):
    if type(obj) in (list, tuple):
        return obj
    else:
        return [obj, ]


def maybe_multirnn(lst):
    if len(lst) == 1:
        return lst[0]
    else:
        return MultiRNNCell(lst)

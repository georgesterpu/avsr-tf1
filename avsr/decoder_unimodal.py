import tensorflow as tf
from tensorflow.contrib import seq2seq
from .cells import build_rnn_layers
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops import array_ops
from .devel import focal_loss, mc_loss


class Seq2SeqUnimodalDecoder(object):
    r"""
    A standard Decoder for Seq2seq models
    """
    def __init__(self,
                 encoder_output,
                 encoder_features_len,
                 labels,
                 labels_length,
                 mode,
                 hparams):
        r"""
        Constructor responsible with the initialisation of the decoder
        Args:
            encoder_output: An `EncoderData` object, holding the encoder output and the final state
            encoder_features_len: A 1D Tensor of shape [batch_size] holding the true length of each
              sequence from the batch
            labels: A 2D Tensor of shape [batch_size, max_label_len] holding the ground truth
              transcriptions. The data type should be tf.int32
            labels_length: A 1D Tensor of shape [batch_size] holding the true length of each
              transcription.
            mode: A Python `String` flag for the `train` or `test` modes
            hparams: A `tf.contrib.training.HParams` object containing necessary hyperparameters.
        """

        self._encoder_output = encoder_output
        self._encoder_features_len = encoder_features_len

        self._labels = labels
        self._labels_len = labels_length

        self._hparams = hparams

        self._mode = mode

        reverse_dict = {v: k for k, v in hparams.unit_dict.items()}

        self._GO_ID = reverse_dict['GO']
        self._EOS_ID = reverse_dict['EOS']
        self._sampling_probability_outputs = hparams.sampling_probability_outputs
        self._vocab_size = len(hparams.unit_dict) - 1  # num unique symbols we expect in the decoder's inputs

        self._global_step = tf.Variable(0, trainable=False, name='global_step')

        self._batch_size, _ = tf.unstack(tf.shape(self._labels))

        # create model
        self._add_special_symbols()
        self._init_embedding()
        self._prepare_attention_memories()
        self._init_decoder()

    def _add_special_symbols(self):
        r"""
        Pads the GO id at the start of each label.
        We assume that the EOS id has already been written during dataset generation.
        """
        _GO_SLICE = tf.ones([self._batch_size, 1], dtype=tf.int32) * self._GO_ID

        self._labels_padded_GO = tf.concat([_GO_SLICE, self._labels], axis=1)

    def _init_embedding(self):
        r"""
        Creates the embedding matrix
        If hparams.vocab_size is non-positive, then we fall back to one-hot encodings
        """
        if self._hparams.embedding_size <= 0:
            self._embedding_matrix = tf.eye(self._vocab_size, dtype=self._hparams.dtype)  # one-hot
        else:
            with tf.variable_scope("embeddings"):

                initialiser = tf.random_uniform_initializer(
                    minval=-1.732 / self._vocab_size,
                    maxval=1.732 / self._vocab_size,
                    dtype=self._hparams.dtype)

                self._embedding_matrix = tf.get_variable(
                    name="embedding_matrix",
                    shape=[self._vocab_size, self._hparams.embedding_size],
                    initializer=initialiser,
                    dtype=self._hparams.dtype,
                    trainable=True if self._mode == 'train' else False,
                )

    def _init_decoder(self):
        r"""
        Builds the decoder blocks: the cells, the initial state, the output projection layer,
        the decoding algorithm, the attention layers and the trainining optimiser
        """

        with tf.variable_scope("Decoder"):

            self._decoder_cells = build_rnn_layers(
                cell_type=self._hparams.cell_type,
                num_units_per_layer=self._hparams.decoder_units_per_layer,
                use_dropout=self._hparams.use_dropout,
                dropout_probability=self._hparams.dropout_probability,
                mode=self._mode,
                dtype=self._hparams.dtype,
            )

            self._construct_decoder_initial_state()

            self._dense_layer = Dense(self._vocab_size,
                                      name='my_dense',
                                      dtype=self._hparams.dtype)

            if self._mode == 'train':
                self._build_decoder_train()
                self._init_optimiser()
            else:
                if self._hparams.decoding_algorithm == 'greedy':
                    self._build_decoder_test_greedy()
                elif self._hparams.decoding_algorithm == 'beam_search':
                    self._build_decoder_test_beam_search()
                else:
                    raise Exception('The only supported algorithms are `greedy` and `beam_search`')

    def _construct_decoder_initial_state(self):
        r"""

        """

        encoder_state = self._encoder_output.final_state

        enc_layers = len(self._hparams.encoder_units_per_layer)
        dec_layers = len(self._hparams.decoder_units_per_layer)

        if enc_layers == 1:
            encoder_state = [encoder_state, ]

        if dec_layers == 1:  # N - 1
            self._decoder_initial_state = encoder_state[-1]
        else:
            if self._hparams.bijective_state_copy is True:  # N - N
                if enc_layers != dec_layers:
                    raise ValueError('The bijective decoder initialisation scheme requires'
                                     'equal number of layers and units in both RNNs')
                self._decoder_initial_state = encoder_state  # list of objects
            else:  # M - N
                self._decoder_initial_state = [encoder_state[-1], ]
                for j in range(dec_layers - 1):
                    zero_state = self._decoder_cells.zero_state(self._batch_size, self._hparams.dtype)
                    self._decoder_initial_state.append(zero_state[j])
                self._decoder_initial_state = tuple(self._decoder_initial_state)

    def _prepare_attention_memories(self):
        r"""
        Optionally processes the memory attended to by the decoder
        """
        self._encoder_memory = self._encoder_output.outputs

    def _create_attention_mechanisms(self, beam_search=False):
        r"""
        Creates a list of attention mechanisms (e.g. seq2seq.BahdanauAttention)
        and also a list of ints holding the attention projection layer size
        Args:
            beam_search: `bool`, whether the beam-search decoding algorithm is used or not
        """
        mechanisms = []
        layer_sizes = []

        if beam_search is True:
            encoder_memory = seq2seq.tile_batch(
                self._encoder_memory, multiplier=self._hparams.beam_width)

            encoder_features_len = seq2seq.tile_batch(
                self._encoder_features_len, multiplier=self._hparams.beam_width)

        else:
            encoder_memory = self._encoder_memory
            encoder_features_len = self._encoder_features_len

        for attention_type in self._hparams.attention_type[0]:

            attention = self._create_attention_mechanism(
                num_units=self._hparams.decoder_units_per_layer[-1],
                memory=encoder_memory,
                memory_sequence_length=encoder_features_len,
                attention_type=attention_type
            )
            mechanisms.append(attention)
            layer_sizes.append(self._hparams.decoder_units_per_layer[-1])

        return mechanisms, layer_sizes

    def _build_decoder_train(self):
        r"""
        Builds the decoder(s) used in training
        """

        self._decoder_train_inputs = tf.nn.embedding_lookup(self._embedding_matrix, self._labels_padded_GO)

        self._basic_decoder_train_outputs, self._final_states, self._final_seq_lens = self._basic_decoder_train()

    def _build_decoder_test_greedy(self):
        r"""
        Builds the greedy test decoder, which feeds the most likely decoded symbol as input for the
        next timestep
        """
        self._helper_greedy = seq2seq.GreedyEmbeddingHelper(
            embedding=self._embedding_matrix,
            start_tokens=tf.tile([self._GO_ID], [self._batch_size]),
            end_token=self._EOS_ID)

        if self._hparams.enable_attention is True:
            cells, initial_state = self._add_attention(decoder_cells=self._decoder_cells, beam_search=False)
        else:
            cells = self._decoder_cells
            initial_state = self._decoder_initial_state

        self._decoder_inference = seq2seq.BasicDecoder(
            cell=cells,
            helper=self._helper_greedy,
            initial_state=initial_state,
            output_layer=self._dense_layer)

        outputs, states, lengths = seq2seq.dynamic_decode(
            self._decoder_inference,
            impute_finished=True,
            swap_memory=False,
            maximum_iterations=self._hparams.max_label_length)

        self.inference_outputs = outputs.rnn_output
        self.inference_predicted_ids = outputs.sample_id

        if self._hparams.write_attention_alignment is True:
            self.attention_summary = self._create_attention_alignments_summary(states, )

    def _build_decoder_test_beam_search(self):
        r"""
        Builds a beam search test decoder
        """
        if self._hparams.enable_attention is True:
            cells, initial_state = self._add_attention(self._decoder_cells, beam_search=True)
        else:  # does the non-attentive beam decoder need tile_batch ?
            cells = self._decoder_cells

            decoder_initial_state_tiled = seq2seq.tile_batch(  # guess so ? it compiles without it too
                self._decoder_initial_state, multiplier=self._hparams.beam_width)
            initial_state = decoder_initial_state_tiled

        self._decoder_inference = seq2seq.BeamSearchDecoder(
            cell=cells,
            embedding=self._embedding_matrix,
            start_tokens=array_ops.fill([self._batch_size], self._GO_ID),
            end_token=self._EOS_ID,
            initial_state=initial_state,
            beam_width=self._hparams.beam_width,
            output_layer=self._dense_layer,
            length_penalty_weight=0.6,
        )

        outputs, states, lengths = seq2seq.dynamic_decode(
            self._decoder_inference,
            impute_finished=False,
            maximum_iterations=self._hparams.max_label_length,
            swap_memory=False)

        self.inference_outputs = outputs.beam_search_decoder_output
        self.inference_predicted_ids = outputs.predicted_ids[:, :, 0]  # return the first beam
        self.inference_predicted_beam = outputs.predicted_ids

    def _create_attention_mechanism(self,
                                    attention_type,
                                    num_units,
                                    memory,
                                    memory_sequence_length):
        r"""
        Instantiates a seq2seq attention mechanism, also setting the _output_attention flag accordingly.

        Warning: if different types of mechanisms are used within the same decoder, this function needs
        to be refactored to return the right output_attention flag for each `AttentionWrapper` object.
        Args:
            attention_type: `String`, one of `bahdanau`, `luong` with optional `normed`, `scaled` or
              `monotonic` prefixes. See code for the precise format.
            num_units: `int`, depth of the query mechanism. See downstream documentation.
            memory: A 3D Tensor [batch_size, Ts, num_features], the attended memory
            memory_sequence_length: A 1D Tensor [batch_size] holding the true sequence lengths
        """

        if attention_type == 'bahdanau':
            attention_mechanism = seq2seq.BahdanauAttention(
                num_units=num_units,
                memory=memory,
                memory_sequence_length=memory_sequence_length,
                normalize=False,
                dtype=self._hparams.dtype
            )
            self._output_attention = False
        elif attention_type == 'normed_bahdanau':
            attention_mechanism = seq2seq.BahdanauAttention(
                num_units=num_units,
                memory=memory,
                memory_sequence_length=memory_sequence_length,
                normalize=True,
                dtype=self._hparams.dtype,
            )
            self._output_attention = False
        elif attention_type == 'normed_monotonic_bahdanau':
            attention_mechanism = seq2seq.BahdanauMonotonicAttention(
                num_units=num_units,
                memory=memory,
                memory_sequence_length=memory_sequence_length,
                normalize=True,
                score_bias_init=-2.0,
                sigmoid_noise=1.0 if self._mode == 'train' else 0.0,
                mode='hard' if self._mode != 'train' else 'parallel',
                dtype=self._hparams.dtype,
            )
            self._output_attention = False
        elif attention_type == 'luong':
            attention_mechanism = seq2seq.LuongAttention(
                num_units=num_units,
                memory=memory,
                memory_sequence_length=memory_sequence_length,
                dtype=self._hparams.dtype,
            )
            self._output_attention = True
        elif attention_type == 'scaled_luong':
            attention_mechanism = seq2seq.LuongAttention(
                num_units=num_units,
                memory=memory,
                memory_sequence_length=memory_sequence_length,
                scale=True,
                dtype=self._hparams.dtype,
            )
            self._output_attention = True
        elif attention_type == 'scaled_monotonic_luong':
            attention_mechanism = seq2seq.LuongMonotonicAttention(
                num_units=num_units,
                memory=memory,
                memory_sequence_length=memory_sequence_length,
                scale=True,
                score_bias_init=-2.0,
                sigmoid_noise=1.0 if self._mode == 'train' else 0.0,
                mode='hard' if self._mode != 'train' else 'parallel',
                dtype=self._hparams.dtype,
            )
            self._output_attention = True
        else:
            raise Exception('unknown attention mechanism')

        return attention_mechanism

    def _create_attention_alignments_summary(self, states):
        r"""
        Generates the alignment images, useful for visualisation/debugging purposes
        """
        attention_alignment = states.alignment_history.stack()

        attention_images = tf.expand_dims(tf.transpose(attention_alignment, [1, 2, 0]), -1)

        # attention_images_scaled = tf.image.resize_images(1-attention_images, (256,128))
        attention_images_scaled = 1 - attention_images

        attention_summary = tf.summary.image("attention_images", attention_images_scaled,
                                             max_outputs=self._hparams.batch_size[1])

        return attention_summary

    def get_predictions(self):
        r"""
        Returns the predictions made by the decoder.
        When beam_search is True, returns the top beam alone
        """
        return self.inference_predicted_ids

    def _init_optimiser(self):
        r"""
        Computes the batch_loss function to be minimised
        """

        self._loss_weights = tf.sequence_mask(
            lengths=self._labels_len,
            dtype=self._hparams.dtype
        )

        if self._hparams.loss_fun is None:
            softmax_loss_fun = None
        elif self._hparams.loss_fun == 'focal_loss':
            softmax_loss_fun = focal_loss
        elif self._hparams.loss_fun == 'mc_loss':
            softmax_loss_fun = mc_loss
        else:
            raise ValueError('Unknown loss function {}'.format(self._hparams.loss_fun))

        self.batch_loss = seq2seq.sequence_loss(
            logits=self._basic_decoder_train_outputs.rnn_output,
            targets=self._labels,
            weights=self._loss_weights,
            softmax_loss_function=softmax_loss_fun,
            average_across_batch=True,
            average_across_timesteps=True)

        reg_loss = 0

        if self._hparams.recurrent_l2_regularisation is not None:
            regularisable_vars = _get_trainable_vars(self._hparams.cell_type)
            reg = tf.contrib.layers.l2_regularizer(scale=self._hparams.recurrent_l2_regularisation)
            reg_loss = tf.contrib.layers.apply_regularization(reg, regularisable_vars)

        if self._hparams.video_processing is not None:
            if 'cnn' in self._hparams.video_processing:
                # we regularise the cnn vars by specifying a regulariser in conv2d
                reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                reg_loss += tf.reduce_sum(reg_variables)

        self.batch_loss = self.batch_loss + reg_loss

        if self._hparams.loss_scaling > 1:
            self.batch_loss *= self._hparams.loss_scaling

        if self._hparams.optimiser == 'Adam':
            optimiser = tf.train.AdamOptimizer(
                learning_rate=self._hparams.learning_rate,
                epsilon=1e-8 if self._hparams.dtype == tf.float32 else 1e-4,
            )
        elif self._hparams.optimiser == 'AdamW':
            from tensorflow.contrib.opt import AdamWOptimizer
            optimiser = AdamWOptimizer(
                learning_rate=self._hparams.learning_rate,
                weight_decay=self._hparams.weight_decay,
                epsilon=1e-8 if self._hparams.dtype == tf.float32 else 1e-4,
            )
        elif self._hparams.optimiser == 'Momentum':
            optimiser = tf.train.MomentumOptimizer(
                learning_rate=self._hparams.learning_rate,
                momentum=0.9,
                use_nesterov=False
            )
        elif self._hparams.optimiser == 'AMSGrad':
            from .AMSGrad import AMSGrad
            optimiser = AMSGrad(
                learning_rate=self._hparams.learning_rate,
                epsilon=1e-8 if self._hparams.dtype == tf.float32 else 1e-4,
            )
        else:
            raise Exception('Unsupported optimiser, try Adam')

        variables = tf.trainable_variables()
        gradients = tf.gradients(self.batch_loss, variables)

        if self._hparams.loss_scaling > 1:
            gradients = [tf.div(grad, self._hparams.loss_scaling) for grad in gradients]

        if self._hparams.clip_gradients is True:
            gradients, _ = tf.clip_by_global_norm(gradients, self._hparams.max_gradient_norm)

        if self._hparams.batch_normalisation is True:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimiser.apply_gradients(

                    zip(gradients, variables), global_step=tf.train.get_global_step())
        else:
            self.train_op = optimiser.apply_gradients(
                zip(gradients, variables))

    def _basic_decoder_train(self):
        r"""
        Builds the standard teacher-forcing training decoder with sampling from previous predictions.
        """

        helper_train = seq2seq.ScheduledEmbeddingTrainingHelper(
            inputs=self._decoder_train_inputs,
            sequence_length=self._labels_len,
            embedding=self._embedding_matrix,
            sampling_probability=self._sampling_probability_outputs,
        )

        if self._hparams.enable_attention is True:
            cells, initial_state = self._add_attention(self._decoder_cells)
        else:
            cells = self._decoder_cells
            initial_state = self._decoder_initial_state

        decoder_train = seq2seq.BasicDecoder(
            cell=cells,
            helper=helper_train,
            initial_state=initial_state,
            output_layer=self._dense_layer,
        )

        outputs, fstate, fseqlen = seq2seq.dynamic_decode(
            decoder_train,
            output_time_major=False,
            impute_finished=True,
            swap_memory=False,

        )

        return outputs, fstate, fseqlen

    def _add_attention(self, decoder_cells, beam_search=False):
        r"""
        Wraps the decoder_cells with an AttentionWrapper
        Args:
            decoder_cells: instances of `RNNCell`
            beam_search: `bool` flag for beam search decoders

        Returns:
            attention_cells: the Attention wrapped decoder cells
            initial_state: a proper initial state to be used with the returned cells
        """
        attention_mechanisms, layer_sizes = self._create_attention_mechanisms(beam_search)

        if beam_search is True:
            decoder_initial_state = seq2seq.tile_batch(
                self._decoder_initial_state, multiplier=self._hparams.beam_width)
        else:
            decoder_initial_state = self._decoder_initial_state

        attention_cells = seq2seq.AttentionWrapper(
            cell=decoder_cells,
            attention_mechanism=attention_mechanisms,
            attention_layer_size=layer_sizes,
            # initial_cell_state=decoder_initial_state,
            alignment_history=False,
            output_attention=self._output_attention,
        )

        attn_zero = attention_cells.zero_state(
            dtype=self._hparams.dtype,
            batch_size=self._batch_size * self._hparams.beam_width if beam_search is True else self._batch_size
        )
        initial_state = attn_zero.clone(
            cell_state=decoder_initial_state
        )

        return attention_cells, initial_state


def _get_trainable_vars(cell_type):
    r"""
    Returns the list of trainable variables associated with the recurrent layers
    """
    cell_type = cell_type.split('_')[0]
    vars_ = [var for var in tf.trainable_variables() if cell_type + '_' in var.name
             and 'bias' not in var.name]
    return vars_


def _project_lstm_state_tuple(state_tuple, num_units):
    r"""
    Concatenates all the `c` and `h` members from a list of `LSTMStateTuple`
      and projects them to a space of dimension `num_units`
    Args:
        state_tuple: a list of `LSTMStateTuple` objects
        num_units: output dimension

    Returns:
        projected_state: a single `LSTMStateTuple` with `c` and `h` of dimension `num_units`
    """
    state_proj_layer = Dense(num_units, name='state_projection', use_bias=False)

    cat_c = tf.concat([state.c for state in state_tuple], axis=-1)
    cat_h = tf.concat([state.h for state in state_tuple], axis=-1)

    proj_c = state_proj_layer(cat_c)
    proj_h = state_proj_layer(cat_h)

    projected_state = tf.contrib.rnn.LSTMStateTuple(c=proj_c, h=proj_h)

    return projected_state

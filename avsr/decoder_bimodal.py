import tensorflow as tf
from tensorflow.contrib import seq2seq
from .cells import build_rnn_layers
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops import array_ops
from tensorflow.contrib.rnn import LSTMStateTuple


class Seq2SeqBimodalDecoder(object):
    def __init__(self,
                 video_output,
                 audio_output,
                 video_features_len,
                 audio_features_len,
                 labels,
                 labels_length,
                 mode,
                 hparams):

        # member variables
        self._video_output = video_output
        self._audio_output = audio_output

        self._video_features_len = video_features_len
        self._audio_features_len = audio_features_len

        self._labels = labels
        self._labels_len = labels_length
        self._hparams = hparams

        self._mode = mode

        reverse_dict = {v: k for k, v in hparams.unit_dict.items()}

        self._GO_ID = reverse_dict['GO']
        self._EOS_ID = reverse_dict['EOS']
        self._sampling_probability_outputs = tf.constant(hparams.sampling_probability_outputs,
                                                         dtype=self._hparams.dtype)
        self._vocab_size = len(hparams.unit_dict) - 1  # excluding END

        self._global_step = tf.Variable(0, trainable=False, name='global_step')

        self._batch_size, _ = tf.unstack(tf.shape(self._labels))

        # create model
        self._infer_num_valid_streams()

        self._add_special_symbols()
        self._init_embedding()
        self._prepare_attention_memories()
        self._init_decoder()

    def _infer_num_valid_streams(self):
        num_streams = 0
        if self._video_output is not None:
            num_streams += 1
        if self._audio_output is not None:
            num_streams += 1

        if num_streams == 0:
            raise Exception('We are totally blind and deaf here...')

        self._num_streams = num_streams

    def _add_special_symbols(self):
        batch_size, sequence_len = tf.unstack(tf.shape(self._labels))
        _GO_SLICE = tf.ones([batch_size, 1], dtype=tf.int32) * self._GO_ID

        self._labels_padded_GO = tf.concat([_GO_SLICE, self._labels], axis=1)

    def _init_embedding(self):
        r"""
        Initialises the lookup matrix that translates a dense representation to a sparse one
        If hparams.vocab_size is non-positive, then we fall back to one-hot encodings
        :return:
        """
        if self._hparams.embedding_size <= 0:
            self._embedding_matrix = tf.eye(self._vocab_size, dtype=self._hparams.dtype)
        else:
            with tf.variable_scope("embeddings"):

                sqrt3 = tf.sqrt(3.0)
                initialiser = tf.random_uniform_initializer(-sqrt3 / self._vocab_size, sqrt3 / self._vocab_size)

                self._embedding_matrix = tf.get_variable(
                    name="embedding_matrix",
                    shape=[self._vocab_size, self._hparams.embedding_size],
                    initializer=initialiser
                )

    def _init_decoder(self):
        r"""
                Instantiates the seq2seq decoder
                :return:
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
                    self._build_decoder_greedy()
                elif self._hparams.decoding_algorithm == 'beam_search':
                    self._build_decoder_beam_search()
                else:
                    raise Exception('The only supported algorithms are `greedy` and `beam_search`')

    def _construct_decoder_initial_state(self):

        if self._video_output is not None:
            video_state = self._video_output.final_state
        else:
            zero_slice = [tf.zeros(shape=tf.shape(self._audio_output.final_state[0].c), dtype=self._hparams.dtype)
                          for _ in range(len(self._audio_output.final_state[0]))]

            video_state = tuple([LSTMStateTuple(c=zero_slice[0], h=zero_slice[1])
                                 for _ in range(len(self._hparams.encoder_units_per_layer))])

        if self._audio_output is not None:
            audio_state = self._audio_output.final_state
        else:
            zero_slice = [tf.zeros(shape=tf.shape(self._video_output.final_state[0].c), dtype=self._hparams.dtype)
                          for _ in range(len(self._video_output.final_state[0]))]
            audio_state = tuple([LSTMStateTuple(c=zero_slice[0], h=zero_slice[1])
                                 for _ in range(len(self._hparams.encoder_units_per_layer))])

        if type(video_state) == tuple:
            final_video_state = video_state[-1]
        else:
            final_video_state = video_state

        if type(audio_state) == tuple:
            final_audio_state = audio_state[-1]
        else:
            final_audio_state = audio_state

        state_tuple = (final_video_state, final_audio_state,)

        self._decoder_initial_state = _project_state_tuple(
            state_tuple, num_units=self._hparams.decoder_units_per_layer[0], cell_type=self._hparams.cell_type)

        dec_layers = len(self._hparams.decoder_units_per_layer)

        if dec_layers > 1:
            self._decoder_initial_state = [self._decoder_initial_state, ]
            zero_state = self._decoder_cells.zero_state(self._batch_size, self._hparams.dtype)
            for j in range(dec_layers - 1):
                self._decoder_initial_state.append(zero_state[j + 1])
            self._decoder_initial_state = tuple(self._decoder_initial_state)

    def _prepare_attention_memories(self):
        if self._video_output is not None:
            self._video_memory = self._video_output.outputs
        else:
            self._video_memory = None

        if self._audio_output is not None:
            self._audio_memory = self._audio_output.outputs
        else:
            self._audio_memory = None

    def _create_attention_mechanisms(self, beam_search=False):

        mechanisms = []
        layer_sizes = []

        if self._video_memory is not None:

            if beam_search is True:
                #  TODO potentially broken, please re-check
                self._video_memory = seq2seq.tile_batch(
                    self._video_memory, multiplier=self._hparams.beam_width)

                self._video_features_len = seq2seq.tile_batch(
                    self._video_features_len, multiplier=self._hparams.beam_width)

            for attention_type in self._hparams.attention_type[0]:

                attention_video = self._create_attention_mechanism(
                    num_units=self._hparams.decoder_units_per_layer[-1],
                    memory=self._video_memory,
                    memory_sequence_length=self._video_features_len,
                    attention_type=attention_type
                )
                mechanisms.append(attention_video)
                layer_sizes.append(self._hparams.decoder_units_per_layer[-1])

        if self._audio_memory is not None:

            if beam_search is True:
                #  TODO potentially broken, please re-check
                self._audio_memory = seq2seq.tile_batch(
                    self._audio_memory, multiplier=self._hparams.beam_width)

                self._audio_features_len = seq2seq.tile_batch(
                    self._audio_features_len, multiplier=self._hparams.beam_width)

            for attention_type in self._hparams.attention_type[1]:
                attention_audio = self._create_attention_mechanism(
                    num_units=self._hparams.decoder_units_per_layer[-1],
                    memory=self._audio_memory,
                    memory_sequence_length=self._audio_features_len,
                    attention_type=attention_type
                )
                mechanisms.append(attention_audio)
                layer_sizes.append(self._hparams.decoder_units_per_layer[-1])

        return mechanisms, layer_sizes

    def _build_decoder_train(self):

        self._labels_embedded = tf.nn.embedding_lookup(self._embedding_matrix, self._labels_padded_GO)

        self._helper_train = seq2seq.ScheduledEmbeddingTrainingHelper(
            inputs=self._labels_embedded,
            sequence_length=self._labels_len,
            embedding=self._embedding_matrix,
            sampling_probability=self._sampling_probability_outputs,
        )

        if self._hparams.enable_attention is True:
            attention_mechanisms, layer_sizes = self._create_attention_mechanisms()

            attention_cells = seq2seq.AttentionWrapper(
                cell=self._decoder_cells,
                attention_mechanism=attention_mechanisms,
                attention_layer_size=layer_sizes,
                initial_cell_state=self._decoder_initial_state,
                alignment_history=False,
                output_attention=self._output_attention,
            )
            batch_size, _ = tf.unstack(tf.shape(self._labels))

            attn_zero = attention_cells.zero_state(
                dtype=self._hparams.dtype, batch_size=batch_size
            )
            initial_state = attn_zero.clone(
                cell_state=self._decoder_initial_state
            )

            cells = attention_cells
        else:
            cells = self._decoder_cells
            initial_state = self._decoder_initial_state

        self._decoder_train = seq2seq.BasicDecoder(
            cell=cells,
            helper=self._helper_train,
            initial_state=initial_state,
            output_layer=self._dense_layer,
        )

        self._decoder_train_outputs, self._final_states, self._final_seq_lens = seq2seq.dynamic_decode(
            self._decoder_train,
            output_time_major=False,
            impute_finished=True,
            swap_memory=False,
        )

    def _build_decoder_greedy(self):

        batch_size, _ = tf.unstack(tf.shape(self._labels))
        self._helper_greedy = seq2seq.GreedyEmbeddingHelper(
            embedding=self._embedding_matrix,
            start_tokens=tf.tile([self._GO_ID], [batch_size]),
            end_token=self._EOS_ID)

        if self._hparams.enable_attention is True:
            attention_mechanisms, layer_sizes = self._create_attention_mechanisms()

            attention_cells = seq2seq.AttentionWrapper(
                cell=self._decoder_cells,
                attention_mechanism=attention_mechanisms,
                attention_layer_size=layer_sizes,
                initial_cell_state=self._decoder_initial_state,
                alignment_history=self._hparams.write_attention_alignment,
                output_attention=self._output_attention
            )
            attn_zero = attention_cells.zero_state(
                dtype=self._hparams.dtype, batch_size=batch_size
            )
            initial_state = attn_zero.clone(
                cell_state=self._decoder_initial_state
            )
            cells = attention_cells
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

        # self._result = outputs, states, lengths
        self.inference_outputs = outputs.rnn_output
        self.inference_predicted_ids = outputs.sample_id

        if self._hparams.write_attention_alignment is True:
            self.attention_summary = self._create_attention_alignments_summary(states, )

    def _build_decoder_beam_search(self):

        batch_size, _ = tf.unstack(tf.shape(self._labels))

        attention_mechanisms, layer_sizes = self._create_attention_mechanisms(beam_search=True)

        decoder_initial_state_tiled = seq2seq.tile_batch(
            self._decoder_initial_state, multiplier=self._hparams.beam_width)

        if self._hparams.enable_attention is True:

            attention_cells = seq2seq.AttentionWrapper(
                cell=self._decoder_cells,
                attention_mechanism=attention_mechanisms,
                attention_layer_size=layer_sizes,
                initial_cell_state=decoder_initial_state_tiled,
                alignment_history=False,
                output_attention=self._output_attention)

            initial_state = attention_cells.zero_state(
                dtype=self._hparams.dtype, batch_size=batch_size * self._hparams.beam_width)

            initial_state = initial_state.clone(
                cell_state=decoder_initial_state_tiled)

            cells = attention_cells
        else:
            cells = self._decoder_cells
            initial_state = decoder_initial_state_tiled

        self._decoder_inference = seq2seq.BeamSearchDecoder(
            cell=cells,
            embedding=self._embedding_matrix,
            start_tokens=array_ops.fill([batch_size], self._GO_ID),
            end_token=self._EOS_ID,
            initial_state=initial_state,
            beam_width=self._hparams.beam_width,
            output_layer=self._dense_layer,
            length_penalty_weight=0.5,
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

        if attention_type == 'bahdanau':
            attention_mechanism = seq2seq.BahdanauAttention(
                num_units=num_units,
                memory=memory,
                memory_sequence_length=memory_sequence_length,
                normalize=False
            )
            self._output_attention = False
        elif attention_type == 'normed_bahdanau':
            attention_mechanism = seq2seq.BahdanauAttention(
                num_units=num_units,
                memory=memory,
                memory_sequence_length=memory_sequence_length,
                normalize=True
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
                mode='hard' if self._mode != 'train' else 'parallel'
            )
            self._output_attention = False
        elif attention_type == 'luong':
            attention_mechanism = seq2seq.LuongAttention(
                num_units=num_units,
                memory=memory,
                memory_sequence_length=memory_sequence_length
            )
            self._output_attention = True
        elif attention_type == 'scaled_luong':
            attention_mechanism = seq2seq.LuongAttention(
                num_units=num_units,
                memory=memory,
                memory_sequence_length=memory_sequence_length,
                scale=True,
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
                mode='hard' if self._mode != 'train' else 'parallel'
            )
            self._output_attention = True
        else:
            raise Exception('unknown attention mechanism')

        return attention_mechanism

    def _create_attention_alignments_summary(self, states):
        attention_alignment = states.alignment_history.stack()

        attention_images = tf.expand_dims(tf.transpose(attention_alignment, [1, 2, 0]), -1)

        # attention_images_scaled = tf.image.resize_images(1-attention_images, (256,128))
        attention_images_scaled = 1 - attention_images

        attention_summary = tf.summary.image("attention_images", attention_images_scaled,
                                             max_outputs=self._hparams.batch_size[1])

        return attention_summary

    def get_predictions(self):
        return self.inference_predicted_ids

    def _init_optimiser(self):
        r"""
            Computes the batch_loss function to be minimised
            :return:
            """

        self._loss_weights = tf.sequence_mask(
            lengths=self._labels_len,
            dtype=self._hparams.dtype
        )

        self.batch_loss = seq2seq.sequence_loss(
            logits=self._decoder_train_outputs.rnn_output,
            targets=self._labels,
            weights=self._loss_weights)

        reg_loss = 0

        if self._hparams.recurrent_l2_regularisation is not None:
            regularisable_vars = _get_trainable_vars(self._hparams.cell_type)
            reg = tf.contrib.layers.l2_regularizer(scale=self._hparams.recurrent_l2_regularisation)
            reg_loss = tf.contrib.layers.apply_regularization(reg, regularisable_vars)

        if self._hparams.video_processing is not None:
            if 'cnn' in self._hparams.video_processing:
                #  we regularise the cnn vars by passing a regulariser in conv2d
                reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                reg_loss += tf.reduce_sum(reg_variables)

        self.batch_loss = self.batch_loss + reg_loss

        if self._hparams.optimiser == 'Adam':
            optimiser = tf.train.AdamOptimizer(learning_rate=self._hparams.learning_rate, epsilon=1e-8)
        elif self._hparams.optimiser == 'Momentum':
            optimiser = tf.train.MomentumOptimizer(
                learning_rate=self._hparams.learning_rate,
                momentum=0.9,
                use_nesterov=False)
        elif self._hparams.optimiser == 'AMSGrad':
            from .AMSGrad import AMSGrad
            optimiser = AMSGrad(
                learning_rate=self._hparams.learning_rate
            )
        else:
            raise Exception('Unsupported Optimiser, try Adam')

        variables = tf.trainable_variables()
        gradients = tf.gradients(self.batch_loss, variables)

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


def _get_trainable_vars(cell_type):
    cell_type = cell_type.split('_')[0]
    vars_ = [var for var in tf.trainable_variables() if cell_type + '_' in var.name
             and 'bias' not in var.name]
    return vars_


def _project_lstm_state_tuple(state_tuple, num_units):

    state_proj_layer = Dense(num_units, name='state_projection', use_bias=False)

    cat_c = tf.concat([state.c for state in state_tuple], axis=-1)
    cat_h = tf.concat([state.h for state in state_tuple], axis=-1)

    proj_c = state_proj_layer(cat_c)
    proj_h = state_proj_layer(cat_h)

    projected_state = tf.contrib.rnn.LSTMStateTuple(c=proj_c, h=proj_h)

    return projected_state


def _project_gru_state_tuple(state_tuple, num_units):

    state_proj_layer = Dense(num_units, name='state_projection', use_bias=False)

    cat = tf.concat([state for state in state_tuple], axis=-1)
    projected_state = state_proj_layer(cat)

    return projected_state


def _project_state_tuple(state_tuple, num_units, cell_type):

    if cell_type == 'lstm':
        state = _project_lstm_state_tuple(state_tuple, num_units)
    else:
        try:
            state = _project_gru_state_tuple(state_tuple, num_units)
        except Exception:
            print('Undefined state fusion behaviour for this cell type: {}'.format(cell_type))
            raise

    return state

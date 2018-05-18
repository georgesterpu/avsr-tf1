import tensorflow as tf
from tensorflow.contrib import seq2seq
from .cells import build_rnn_layers
from .attention import create_attention_mechanism
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
                 hparams,
                 gpu_id):

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

        # create model
        self._infer_num_valid_streams()
        if self._hparams.label_skipping is True and mode == 'train':
            # self._labels_len2 = self._labels_len
            self._labels, self._labels_len = self._label_skipping()
        self._add_special_symbols()
        self._init_embedding()
        self._construct_decoder_initial_state()
        self._prepare_attention_memories()
        self._init_decoder(gpu_id)

    def _infer_num_valid_streams(self):
        num_streams = 0
        if self._video_output is not None:
            num_streams += 1
        if self._audio_output is not None:
            num_streams += 1

        if num_streams == 0:
            raise Exception('We are totally blind and deaf here...')

        self._num_streams = num_streams

    def _label_skipping(self):

        bs, ts = tf.unstack(tf.shape(self._labels))

        def expand_fnc(x):
            import numpy as np
            lst = np.array((), dtype=np.int32)
            for idx in np.arange(np.size(x)):
                r = np.arange(x[idx], dtype=np.int32)
                lst = np.append(lst, r)
            return lst

        bool_table = tf.not_equal(self._labels, self._global_step % 11 + 1)
        good_indices = tf.cast(tf.where(bool_table), dtype=tf.int32)
        good_values = tf.gather_nd(self._labels, good_indices)
        y1, y2, cnt = tf.unique_with_counts(good_indices[:, 0])

        newidx = tf.py_func(expand_fnc, [cnt], tf.int32)
        pair = tf.stack((good_indices[:, 0], newidx), axis=1)

        added_elems = ts - cnt
        new_labels = tf.scatter_nd(pair, good_values, tf.shape(self._labels))
        new_labels_lens = self._labels_len - added_elems

        return new_labels, new_labels_lens

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

    def _init_decoder(self, gpu_id=0):
        r"""
                Instantiates the seq2seq decoder
                :return:
                """

        with tf.variable_scope("Decoder"):

            with tf.device('/gpu:{}'.format(gpu_id)):
                self._decoder_cells = build_rnn_layers(
                    cell_type=self._hparams.cell_type,
                    num_units_per_layer=self._hparams.decoder_units_per_layer,
                    use_dropout=self._hparams.use_dropout,
                    dropout_probability=self._hparams.dropout_probability,
                    mode=self._mode,
                    base_gpu=gpu_id  # decoder runs on a single GPU
                )
    
                self._dense_layer = Dense(self._vocab_size,
                                          name='my_dense',
                                          dtype=self._hparams.dtype)

            if self._mode == 'train':
                self._build_decoder_train()
                with tf.device('/gpu:{}'.format(gpu_id)):
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
            try:
                #if it is an AttentionWrapperState, need to get cell_state
                video_state = video_state.cell_state
            except:
                pass
        else:
            zero_slice = [tf.zeros(shape=tf.shape(self._audio_output.final_state[0].c), dtype=self._hparams.dtype)
                          for _ in range(len(self._audio_output.final_state[0]))]

            video_state = tuple([LSTMStateTuple(c=zero_slice[0], h=zero_slice[1]) for _ in range(len(self._hparams.encoder_units_per_layer))])

        if self._audio_output is not None:
            audio_state = self._audio_output.final_state
            try:
                #if it is an AttentionWrapperState, need to get cell_state
                audio_state = audio_state.cell_state
            except:
                pass

        else:
            zero_slice = [tf.zeros(shape=tf.shape(self._video_output.final_state[0].c), dtype=self._hparams.dtype)
                           for _ in range(len(self._video_output.final_state[0])) ]
            audio_state = tuple([LSTMStateTuple(c=zero_slice[0], h=zero_slice[1]) for _ in range(len(self._hparams.encoder_units_per_layer))])

        state_tuples = []

        if len(self._hparams.encoder_units_per_layer) == 1:
            video_state = (video_state, )
            audio_state = (audio_state, )

        for i in range(len(self._hparams.encoder_units_per_layer)):
            if isinstance(video_state[i], LSTMStateTuple):
                cat_c = tf.concat((video_state[i].c, audio_state[i].c), axis=-1)
                cat_h = tf.concat((video_state[i].h, audio_state[i].h), axis=-1)
                state_tuples.append(LSTMStateTuple(c=cat_c, h=cat_h))
            else:
                video_state_c, video_state_h = tf.split(video_state[i],
                                                        num_or_size_splits=2,
                                                        axis=-1)
                audio_state_c, audio_state_h = tf.split(audio_state[i],
                                                        num_or_size_splits=2,
                                                        axis=-1)
                cat_c = tf.concat((video_state_c, audio_state_c), -1)
                cat_h = tf.concat((video_state_h, audio_state_h), -1) 
                state_tuples.append(tf.concat((cat_c, cat_h), -1))

        state_tuples = tuple(state_tuples)
            

        if len(self._hparams.encoder_units_per_layer) == 1:
            state_tuples = state_tuples[0]

        if len(self._hparams.decoder_units_per_layer) != len(self._hparams.encoder_units_per_layer):
            ## option 1
            # self._decoder_initial_state = state_tuples[-1]
            # make sure that encoder_units[-1] == decoder_units[0]
            # to make the N layer encoder -> 1 layer decoder arch work

            ## option 2

            if isinstance(state_tuples[0], LSTMStateTuple):
                self._decoder_initial_state = _project_lstm_state_tuple(
                    state_tuples, num_units=self._hparams.decoder_units_per_layer[0])
            else:
                self._decoder_initial_state = _project_state(
                    state_tuples, num_units=self._hparams.decoder_units_per_layer[0])
        else:
            self._decoder_initial_state = state_tuples
            # make sure that encoder_units[i] == decoder_units[i] for i in num_layers
            # to make the N layer encoder -> N layer decoder arch work

    def _prepare_attention_memories(self):
        if self._video_output is not None:
            self._video_memory = self._video_output.outputs
        else:
            self._video_memory = None

        if self._audio_output is not None:
            self._audio_memory = self._audio_output.outputs
        else:
            self._audio_memory = None

    def _create_attention_mechanisms(self, beam_search=False, num_units=0):

        mechanisms = []
        layer_sizes = []
        
        num_units = (num_units if num_units > 0 else 
                    self._hparams.decoder_units_per_layer[-1])

        if self._video_memory is not None:

            if beam_search is True:
                ## TODO potentially broken, please re-check
                self._video_memory = seq2seq.tile_batch(
                    self._video_memory, multiplier=self._hparams.beam_width)

                self._video_features_len = seq2seq.tile_batch(
                    self._video_features_len, multiplier=self._hparams.beam_width)

            for attention_type in self._hparams.attention_type[0]:

                #XXX: is -1 the right assumption? it's the last added layer
                #i.e. the topmost
                attention_video, self._output_attention = create_attention_mechanism(
                    num_units=num_units,
                    memory=self._video_memory,
                    memory_sequence_length=self._video_features_len,
                    attention_type=attention_type
                )
                mechanisms.append(attention_video)
                layer_sizes.append(num_units)

        if self._audio_memory is not None:

            if beam_search is True:
                ## TODO potentially broken, please re-check
                self._audio_memory = seq2seq.tile_batch(
                    self._audio_memory, multiplier=self._hparams.beam_width)

                self._audio_features_len = seq2seq.tile_batch(
                    self._audio_features_len, multiplier=self._hparams.beam_width)

            for attention_type in self._hparams.attention_type[1]:
                attention_audio, self._output_attention = create_attention_mechanism(
                    num_units=num_units,
                    memory=self._audio_memory,
                    memory_sequence_length=self._audio_features_len,
                    attention_type=attention_type
                )
                mechanisms.append(attention_audio)
                layer_sizes.append(num_units)

        return mechanisms, layer_sizes

    def _build_decoder_train(self):

        with tf.device('/gpu:{}'.format(self._hparams.num_gpus % 1)):
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
                swap_memory=self._hparams.swap_memory,
            )
            
    def _build_decoder_split_train(self):

        with tf.device('/gpu:{}'.format(self._hparams.num_gpus % 1)):
            self._labels_embedded = tf.nn.embedding_lookup(self._embedding_matrix, self._labels_padded_GO)
    
            self._helper_train = seq2seq.ScheduledEmbeddingTrainingHelper(
                inputs=self._labels_embedded,
                sequence_length=self._labels_len,
                embedding=self._embedding_matrix,
                sampling_probability=self._sampling_probability_outputs,
            )
            num_units_att = self.decoder_units_per_layer_am
            if self._hparams.enable_attention is True:
                #attention goes into the lowest acoustic layer
                attention_mechanisms, layer_sizes = self._create_attention_mechanisms(num_units=num_units_att)
    
                attention_cells = seq2seq.AttentionWrapper(
                    cell=self._decoder_am_cells,
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
                swap_memory=self._hparams.swap_memory,
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
            swap_memory=self._hparams.swap_memory,
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



    def get_predictions(self):
        return self.inference_predicted_ids

    def _init_optimiser(self):
        r"""
            Computes the batch_loss function to be minimised
            :return:
            """
        # emb = tf.constant([0.0, 3.17, 0.64, 1.70, 14.28, 3.44, 8.33, 5.52, 2.29, 0.31, 0.47, 2.06, 1.40, 14.28, 14.28])
        # def elem_iter(y): return tf.nn.embedding_lookup(emb, y)
        # def row_iter(x): return elem_iter(x)
        # self._loss_weights2 = tf.map_fn(row_iter, self._labels, dtype=tf.float32)

        self._loss_weights = tf.sequence_mask(
            lengths=self._labels_len,
            dtype=self._hparams.dtype
        )
        # self._loss_weights = tf.multiply(self._loss_weights, self._loss_weights2)

        if self._hparams.label_skipping is True and self._mode == 'train':
            # diff = tf.shape(self._labels)[-1] - tf.shape(self._loss_weights)[-1]
            self._labels = self._labels[:,:tf.shape(self._loss_weights)[-1]]

        # self._labels = tf.Print(self._labels, [diff], summarize=1000)
        self.batch_loss = seq2seq.sequence_loss(
            logits=self._decoder_train_outputs.rnn_output,
            targets=self._labels,
            weights=self._loss_weights)

        reg_loss = 0

        if self._hparams.recurrent_regularisation is not None:
            regularisable_vars = self._get_trainable_vars(self._hparams.cell_type)
            reg = tf.contrib.layers.l2_regularizer(scale=self._hparams.recurrent_regularisation)
            reg_loss = tf.contrib.layers.apply_regularization(reg, regularisable_vars)

        # if 'cnn' in self._hparams.video_processing:
        if self._hparams.video_processing is not None:
            if 'cnn' in self._hparams.video_processing:
                # !!we regularise the cnn vars by specifying a regulariser in conv2d!!
                reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                reg_loss += tf.reduce_sum(reg_variables)

        self.batch_loss = self.batch_loss + reg_loss

        if self._hparams.use_ctc is True:
            projected_encoder_outputs = self._dense_layer.apply(self._encoder_outputs)
            # self.labels_sparse = ctc_label_dense_to_sparse(self._labels, self._labels_len)
            idx = tf.where(tf.not_equal(self._labels, 0))
            self.labels_sparse = tf.SparseTensor(idx, tf.gather_nd(self._labels, idx),
                                                 tf.shape(self._labels, out_type=tf.int64))

            ctc_loss = tf.reduce_mean(tf.nn.ctc_loss(
                labels=self.labels_sparse,
                inputs=projected_encoder_outputs,
                sequence_length=self._inputs_len,
                time_major=False,
            ))

            self.batch_loss = 0.8 * self.batch_loss + 0.2 * ctc_loss

        self._lr_decayed = self._hparams.learning_rate
        if self._hparams.lr_decay == 'staircase':
            self._lr_decayed = tf.train.exponential_decay(
                learning_rate=self._lr_decayed,
                global_step=self._global_step,
                decay_steps=992500,
                decay_rate=0.1,
                staircase=True
            )
        else:
            raise Exception('unsupported learning rate decay mode')
        
        
        if self._hparams.optimiser == 'Adam':
            optimiser = tf.train.AdamOptimizer(learning_rate=self._lr_decayed, epsilon=1e-8)
        elif self._hparams.optimiser == 'Momentum':
            optimiser = tf.train.MomentumOptimizer(
                learning_rate=self._lr_decayed,
                momentum=0.9,
                use_nesterov=False)
        elif self._hparams.optimiser == 'AMSGrad':
            from .AMSGrad import AMSGrad
            optimiser = AMSGrad(
                learning_rate=self._lr_decayed
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


    def _get_trainable_vars(self, cell_type):
        cell_type = cell_type.split('_')[0]
        vars = [var for var in tf.trainable_variables() if cell_type + '_' in var.name
                and not 'bias' in var.name]
        return vars

def _project_lstm_state_tuple(state_tuple, num_units):

    state_proj_layer = Dense(num_units,
                            name='state_projection',
                            use_bias=False,
                            )

    cat_c = tf.concat([state.c for state in state_tuple], axis=-1)
    cat_h = tf.concat([state.h for state in state_tuple], axis=-1)

    proj_c = state_proj_layer(cat_c)
    proj_h = state_proj_layer(cat_h)

    projected_state = tf.contrib.rnn.LSTMStateTuple(c=proj_c, h=proj_h)

    return projected_state

def _project_state(state, num_units):

    state_proj_layer = Dense(num_units,
                            name='state_projection',
                            use_bias=False,
                            )

    state_c, state_h = tf.split(state,
                                num_or_size_splits=2,
                                axis=-1)

    proj_c = state_proj_layer(state_c)
    proj_h = state_proj_layer(state_h)

    projected_state = tf.concat((proj_c, proj_h), -1)

    return projected_state

def _build_rnn_layers_split(cell_type,
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
        
        if layer == 1:
            _creat

        cell_list.append(cell)

    if len(cell_list) == 1:
        return cell_list[0]
    else:
        return MultiRNNCell(cell_list), 
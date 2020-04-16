import tensorflow as tf
from .io_utils import create_unit_dict, make_iterator_from_label_record, make_iterator_from_text_dataset
import collections
from .io_utils import BatchedData
from tensorflow.contrib import seq2seq
from .cells import build_rnn_layers
from tensorflow.python.layers.core import Dense
from os import path, makedirs
import time


class Model(collections.namedtuple("Model", ("data", "model", "initializer", "batch_size"))):
    pass


class LM(object):
    def __init__(
            self,
            unit,
            unit_file=None,
            labels_train_record=None,
            labels_test_record=None,
            text_dataset=None,
            batch_size=(64, 64),
            cell_type='lstm',
            recurrent_l2_regularisation=0.0001,
            decoder_units_per_layer=(256,),
            use_dropout=True,
            decoder_dropout_probability=(0.9, 0.9, 0.9),
            embedding_size=128,
            sampling_probability_outputs=0.1,
            optimiser='Adam',
            learning_rate=0.001,
            clip_gradients=True,
            max_gradient_norm=1.0,
            precision='float32',
            required_grahps=('train', 'eval'),
            **kwargs,
            ):

        self._unit = unit
        self._unit_dict = create_unit_dict(unit_file=unit_file)

        self._labels_train_record = labels_train_record
        self._labels_test_record = labels_test_record
        self._text_dataset = text_dataset

        self._required_graphs = required_grahps

        self._hparams = tf.contrib.training.HParams(
            unit_dict=self._unit_dict,
            unit_file=unit_file,
            vocab_size=len(self._unit_dict),
            batch_size=batch_size,
            max_label_length={'viseme': 65, 'phoneme': 70, 'character': 100}[unit],  # max lens from tcdtimit
            cell_type=cell_type,
            recurrent_l2_regularisation=None if optimiser == 'AdamW' else recurrent_l2_regularisation,
            decoder_units_per_layer=decoder_units_per_layer,
            use_dropout=use_dropout,
            decoder_dropout_probability=decoder_dropout_probability,
            embedding_size=embedding_size,
            sampling_probability_outputs=sampling_probability_outputs,
            optimiser=optimiser,
            learning_rate=learning_rate,
            clip_gradients=clip_gradients,
            max_gradient_norm=max_gradient_norm,
            dtype=tf.float16 if precision == 'float16' else tf.float32,
            kwargs=kwargs,
        )

        self._create_graphs()
        self._create_models()
        self._create_sessions()
        self._initialize_sessions()

    def __del__(self):
        if 'train' in self._required_graphs:
            self._train_session.close()
        if 'eval' in self._required_graphs:
            self._evaluate_session.close()

    def _create_graphs(self):
        if 'train' in self._required_graphs:
            self._train_graph = tf.Graph()
        if 'eval' in self._required_graphs:
            self._evaluate_graph = tf.Graph()

    def _create_models(self):
        if 'train' in self._required_graphs:
            self._train_model = self._make_model(
                graph=self._train_graph,
                mode='train',
                batch_size=self._hparams.batch_size[0])

        if 'eval' in self._required_graphs:
            self._evaluate_model = self._make_model(
                graph=self._evaluate_graph,
                mode='evaluate',
                batch_size=self._hparams.batch_size[1])

    def _create_sessions(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        if 'train' in self._required_graphs:
            self._train_session = tf.Session(graph=self._train_graph, config=config)
        if 'eval' in self._required_graphs:
            self._evaluate_session = tf.Session(graph=self._evaluate_graph, config=config)

        self.sess_opts = {}

    def _initialize_sessions(self):
        if 'train' in self._required_graphs:
            for initialiser in self._train_model.initializer:
                self._train_session.run(initialiser)
        if 'eval' in self._required_graphs:
            self._evaluate_session.run(self._evaluate_model.initializer)

    def _make_model(self, graph, mode, batch_size):
        with graph.as_default():

            label_data = self._fetch_data(mode, batch_size)

            model = LanguageModel(
                label_data=label_data,
                mode=mode,
                hparams=self._hparams
            )

            global_vars_initializer = tf.global_variables_initializer()
            tables_initialiser = tf.tables_initializer()

            # Returning the original data, not the processed features
            return Model(data=label_data,
                         model=model,
                         initializer=[global_vars_initializer, tables_initialiser],
                         batch_size=batch_size)

    def _fetch_data(self, mode, batch_size):

        if self._text_dataset is not None:
            label_iterator = make_iterator_from_text_dataset(
                text_dataset=self._text_dataset,
                batch_size=batch_size,
                shuffle=True if mode == 'train' else False,
                unit_dict=self._hparams.unit_dict,
                bucket_width=30,
            )
        else:
            label_iterator = make_iterator_from_label_record(
                    label_record=self._labels_train_record if mode == 'train' else self._labels_test_record,
                    batch_size=batch_size,
                    shuffle=True if mode == 'train' else False,
                    unit_dict=self._hparams.unit_dict,
                    reverse_input=False,
                    bucket_width=30,
                )

        label_data = self._parse_iterator(label_iterator)

        return label_data

    def _parse_iterator(self, iterator):
        labels = tf.cast(iterator.labels, tf.int32, name='labels')
        labels_length = tf.cast(iterator.labels_length, tf.int32, name='labels_len')

        return BatchedData(
            inputs=iterator.inputs,
            inputs_length=iterator.inputs_length,
            inputs_filenames=iterator.inputs_filenames,
            labels=labels,  #
            labels_length=labels_length,  #
            labels_filenames=iterator.labels_filenames,
            iterator_initializer=iterator.iterator_initializer,
            payload=iterator.payload)

    def train(self,
              logfile,
              num_epochs=400,
              try_restore_latest_checkpoint=False
              ):

        checkpoint_dir = path.join('checkpoints', path.split(logfile)[-1])
        checkpoint_path = path.join(checkpoint_dir, 'checkpoint.ckp')
        makedirs(path.dirname(checkpoint_dir), exist_ok=True)
        makedirs(path.dirname(logfile), exist_ok=True)

        last_epoch = 0
        if try_restore_latest_checkpoint is True:
            try:
                latest_ckp = tf.train.latest_checkpoint(checkpoint_dir)
                last_epoch = int(latest_ckp.split('-')[-1])
                self._train_model.model.saver.restore(
                    sess=self._train_session,
                    save_path=latest_ckp, )
                print('Restoring checkpoint from epoch {}\n'.format(last_epoch))
            except Exception:
                print('Could not restore from checkpoint, training from scratch!\n')

        f = open(logfile, 'a')

        for current_epoch in range(1, num_epochs):
            epoch = last_epoch + current_epoch

            self._train_session.run([self._train_model.data.iterator_initializer])
            sum_loss = 0
            batches = 0

            start = time.time()

            try:
                while True:
                    out = self._train_session.run([self._train_model.model.train_op,
                                                   self._train_model.model.batch_loss,
                                                   self._train_model.model.average_log_likelihoods,
                                                   ], **self.sess_opts)

                    sum_loss += out[1]
                    print('batch: {}, batch loss: {}'.format(batches, out[1]))
                    batches += 1

            except tf.errors.OutOfRangeError:
                pass

            print('epoch time: {}'.format(time.time() - start))
            f.write('Average batch_loss as epoch {} is {}\n'.format(epoch, sum_loss / batches))
            f.flush()

            if epoch % 1 == 0:
                save_path = self._train_model.model.saver.save(
                    sess=self._train_session,
                    save_path=checkpoint_path,
                    global_step=epoch,
                )

                # self.evaluate(save_path, epoch)

    def evaluate(self,
                 checkpoint_path,
                 epoch=None,
                 ):
        self._evaluate_model.model.saver.restore(
            sess=self._evaluate_session,
            save_path=checkpoint_path
        )
        self._evaluate_session.run([self._evaluate_model.data.iterator_initializer])
        session_outputs = [self._evaluate_model.model.average_log_likelihoods,
                           self._evaluate_model.data.labels_filenames,
                           ]

        likelihoods_dict = {}

        while True:
            try:
                out = self._evaluate_session.run(session_outputs)

                batch_likelihoods = out[0]
                batch_filenames = out[1]

                for element in range(len(batch_filenames)):
                    filename = batch_filenames[element].decode('utf-8')
                    likelihood = batch_likelihoods[element]
                    likelihoods_dict[filename] = likelihood

            except tf.errors.OutOfRangeError:
                break

        outdir = path.join('predictions', path.split(path.split(checkpoint_path)[0])[-1])
        makedirs(outdir, exist_ok=True)
        outfile = path.join(outdir, 'predicted_epoch_{}.mlf'.format(epoch))

        with open(outfile, 'w') as f:
            dict_to_str = ''.join(['{} {}\n'.format(k, v) for (k, v) in likelihoods_dict.items()])
            f.write(dict_to_str)


class LanguageModel(object):
    def __init__(self,
                 label_data,
                 mode,
                 hparams):

        self._mode = mode
        self._hparams = hparams

        self._labels = label_data.labels
        self._labels_length = label_data.labels_length

        reverse_dict = {v: k for k, v in hparams.unit_dict.items()}
        self._GO_ID = reverse_dict['GO']
        self._EOS_ID = reverse_dict['EOS']
        self._sampling_probability_outputs = hparams.sampling_probability_outputs
        self._vocab_size = len(hparams.unit_dict) - 1

        self._global_step = tf.Variable(0, trainable=False, name='global_step')
        self._batch_size, _ = tf.unstack(tf.shape(self._labels))

        self._make_decoder()

        if mode == 'train':
            self._init_optimiser()

        self._init_saver()

    def _make_decoder(self):
        self._add_special_symbols()
        self._init_embedding()
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
        with tf.variable_scope("Decoder"):
            self._decoder_cells = build_rnn_layers(
                cell_type=self._hparams.cell_type,
                num_units_per_layer=self._hparams.decoder_units_per_layer,
                use_dropout=self._hparams.use_dropout,
                dropout_probability=self._hparams.decoder_dropout_probability,
                mode=self._mode,
                dtype=self._hparams.dtype,
            )

            self._decoder_initial_state = self._decoder_cells.zero_state(
                batch_size=self._batch_size, dtype=self._hparams.dtype)

            self._dense_layer = Dense(
                self._vocab_size,
                name='my_dense',
                dtype=self._hparams.dtype)

            self._build_decoder_train()  # used for both training and evaluation

    def _build_decoder_train(self):
        self._decoder_train_inputs = tf.nn.embedding_lookup(self._embedding_matrix, self._labels_padded_GO)

        if self._mode == 'train':
            sampler = seq2seq.ScheduledEmbeddingTrainingHelper(
                inputs=self._decoder_train_inputs,
                sequence_length=self._labels_length,
                embedding=self._embedding_matrix,
                sampling_probability=self._sampling_probability_outputs,
            )
        else:
            sampler = seq2seq.TrainingHelper(
                inputs=self._decoder_train_inputs,
                sequence_length=self._labels_length,
            )

        cells = self._decoder_cells

        decoder_train = seq2seq.BasicDecoder(
            cell=cells,
            helper=sampler,
            initial_state=self._decoder_initial_state,
            output_layer=self._dense_layer,
        )

        outputs, _, _ = seq2seq.dynamic_decode(
            decoder_train,
            output_time_major=False,
            impute_finished=True,
            swap_memory=False,
        )

        logits = outputs.rnn_output
        self.decoder_train_outputs = logits
        self.average_log_likelihoods = self._compute_likelihood(logits)
        print('')

    def _compute_likelihood(self, logits):
        weights = tf.sequence_mask(
            lengths=self._labels_length,
            dtype=self._hparams.dtype
        )
        log_likelihoods = seq2seq.sequence_loss(
            logits=logits,
            targets=self._labels,
            weights=weights,
            softmax_loss_function=None,
            average_across_batch=False,
            average_across_timesteps=True)

        return log_likelihoods

    def _init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=5)

    def _init_optimiser(self):

        self.current_lr = self._hparams.learning_rate

        self._loss_weights = tf.sequence_mask(
            lengths=self._labels_length,
            dtype=self._hparams.dtype
        )

        self.batch_loss = seq2seq.sequence_loss(
            logits=self.decoder_train_outputs,
            targets=self._labels,
            weights=self._loss_weights,
            softmax_loss_function=None,
            average_across_batch=True,
            average_across_timesteps=True)

        self.reg_loss = 0

        if self._hparams.recurrent_l2_regularisation is not None:
            regularisable_vars = _get_trainable_vars(self._hparams.cell_type)
            reg = tf.contrib.layers.l2_regularizer(scale=self._hparams.recurrent_l2_regularisation)
            self.reg_loss = tf.contrib.layers.apply_regularization(reg, regularisable_vars)

        self.batch_loss = self.batch_loss + self.reg_loss

        if self._hparams.optimiser == 'Adam':
            optimiser = tf.train.AdamOptimizer(
                learning_rate=self.current_lr,
                epsilon=1e-8 if self._hparams.dtype == tf.float32 else 1e-4,
            )
        elif self._hparams.optimiser == 'AdamW':
            from tensorflow.contrib.opt import AdamWOptimizer
            optimiser = AdamWOptimizer(
                learning_rate=self.current_lr,
                weight_decay=self._hparams.weight_decay,
                epsilon=1e-8 if self._hparams.dtype == tf.float32 else 1e-4,
            )
        elif self._hparams.optimiser == 'Momentum':
            optimiser = tf.train.MomentumOptimizer(
                learning_rate=self.current_lr,
                momentum=0.9,
                use_nesterov=False
            )
        else:
            raise Exception('Unsupported optimiser, try Adam')

        variables = tf.trainable_variables()
        gradients = tf.gradients(self.batch_loss, variables)

        if self._hparams.clip_gradients is True:
            gradients, _ = tf.clip_by_global_norm(gradients, self._hparams.max_gradient_norm)

        self.train_op = optimiser.apply_gradients(
            grads_and_vars=zip(gradients, variables),
            global_step=tf.train.get_global_step())


def _get_trainable_vars(cell_type):
    r"""
    Returns the list of trainable variables associated with the recurrent layers
    """
    cell_type = cell_type.split('_')[0]
    vars_ = [var for var in tf.trainable_variables() if cell_type + '_' in var.name
             and 'bias' not in var.name]
    return vars_

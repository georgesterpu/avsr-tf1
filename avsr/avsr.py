import tensorflow as tf
import collections
from .io_utils import make_iterator_from_one_record, create_unit_dict, make_iterator_from_two_records
from .video import cnn_layers
from .audio import process_audio
from .seq2seq import Seq2SeqModel
import time
from os import makedirs, path
from .utils import compute_wer, write_sequences_to_labelfile


class Data(collections.namedtuple("Data", ("inputs", "inputs_length", "inputs_filenames",
                                           "labels", "labels_length", "labels_filenames",
                                           "iterator_initializer"))):
    pass


class Model(collections.namedtuple("Model", ("data", "model", "initializer", "batch_size"))):
    pass


class AVSR(object):
    def __init__(self,
                 unit,
                 unit_file=None,
                 video_processing=None,
                 video_train_record=None,
                 video_test_record=None,
                 audio_processing=None,
                 audio_train_record=None,
                 audio_test_record=None,
                 labels_train_record=None,
                 labels_test_record=None,
                 batch_size=(48, 64),
                 cnn_filters=(8, 16, 32, 64),
                 cnn_dense_units=128,
                 batch_normalisation=True,
                 instance_normalisation=False,
                 input_dense_layers=(0,),
                 architecture='unimodal',
                 encoder_type='unidirectional',
                 highway_encoder=False,
                 residual_encoder=False,
                 cell_type='lstm',
                 recurrent_l2_regularisation=0.0001,
                 weight_decay=0.0001,
                 encoder_units_per_layer=((256, 256, 256), (256, 256, 256)),
                 decoder_units_per_layer=(256,),
                 enable_attention=True,
                 attention_type=(('scaled_luong',)*1, ('scaled_luong',)*1),
                 use_dropout=True,
                 dropout_probability=(0.9, 0.9, 0.9),
                 embedding_size=128,
                 sampling_probability_outputs=0.1,
                 decoding_algorithm='beam_search',
                 beam_width=10,
                 optimiser='Adam',
                 learning_rate=0.001,
                 lr_decay=None,
                 loss_fun=None,
                 clip_gradients=True,
                 max_gradient_norm=1.0,
                 num_gpus=1,
                 write_attention_alignment=False,
                 precision='float32',
                 profiling=False,
                 ):
        r"""

        Args:
            unit: A string that represents the decoded linguistic unit, one of ('phoneme', 'viseme', 'character')
            unit_file: Path to a file storing the unit vocabulary, having one unit per line
            video_processing: Visual CNN front-end,
                one of ('features', 'resnet_cnn', '2dconv_cnn', '3dconv_cnn') or None
            video_train_record: Path to a training .tfrecord file of visual examples
            video_test_record: Path to a testing .tfrecord file of visual examples
            audio_processing: Acoustic front-end, one of ('features', 'wav') or None.
            audio_train_record: Path to a training .tfrecord file of acoustic examples
            audio_test_record: Path to a testing .tfrecord file of acoustic examples
            labels_train_record: Path to a training .tfrecord file of ground-truth transcriptions
            labels_test_record: Path to a training .tfrecord file of ground-truth transcriptions
            batch_size: Number of training examples/full sentences to be processed at once
            cnn_filters: List holding the number of convolutional filters, one list element per CNN layer
            cnn_dense_units: Output size of the CNN front-end, defines the visual feature size
            batch_normalisation: Boolean
            input_dense_layers: List holding the number of units of the optional fully-connected layers added after the
                feature extraction stage, one list element per layer
            architecture: One of ('unimodal', 'bimodal', 'av_align')
            encoder_type: RNN encoder type, one of ('unidirectional', 'bidirectional')
            highway_encoder: Boolean, optionally adds highway connections in the RNNs
            residual_encoder: Boolean, optionally adds residual connections in the RNNs
            cell_type: RNN cell type, one of ('lstm', 'gru'). See cells.py for more variants.
            recurrent_l2_regularisation: Boolean, L2 weight loss
            weight_decay: Float, Weight decay used by the AdamW optimiser.
            encoder_units_per_layer: List holding the number of RNN units in each encoding layer
            decoder_units_per_layer: List holding the number of RNN units in each decoding layer
            enable_attention: Boolean
            attention_type: One of ('luong', 'scaled_luong', 'bahdanau', 'normed_bahadau', 'scaled_monotonic_luong')
            use_dropout: Boolean
            dropout_probability: RNN dropout, list of 3 floats representing the keep probability of the
                (input, state, output) units respectively
            embedding_size: Linguistic unit embedding size, 0 means one-hot encodings, use a positive int to learn an
                embedding matrix of size [vocabulary_size, embedding_size]
            sampling_probability_outputs: Probability of the current decoding prediction id to be used as next input
                instead of the ground truth unit id
            decoding_algorithm: One of ('greedy', 'beam_search')
            beam_width: Controls the beam size when using the `beam_search` algorithm
            optimiser: One of ('Adam', 'AMSGrad')
            learning_rate: Learning algorithm constant
            loss_fun: Loss function. Use None for the default cross-entropy sequence loss, or one of
                ('focal_loss', 'mc_loss) for alternative loss functions
            clip_gradients: Boolean
            max_gradient_norm: Gradient clipping constant, used when gradient clipping is enabled
            num_gpus: deprecated
            write_attention_alignment: Generates alignment images of the attention weights for visualisation purposes
            precision: One of ('float32', 'float16')
            profiling: Generates a runtime profile
        """
        self._unit = unit
        self._unit_dict = create_unit_dict(unit_file=unit_file)

        self._video_processing = video_processing
        self._video_train_record = video_train_record
        self._video_test_record = video_test_record
        self._audio_processing = audio_processing
        self._audio_train_record = audio_train_record
        self._audio_test_record = audio_test_record
        self._labels_train_record = labels_train_record
        self._labels_test_record = labels_test_record

        self._write_attention_alignment = write_attention_alignment

        self._hparams = tf.contrib.training.HParams(
            unit_dict=self._unit_dict,
            unit_file=unit_file,
            vocab_size=len(self._unit_dict),
            batch_size=batch_size,
            video_processing=video_processing,
            audio_processing=audio_processing,
            max_label_length={'viseme': 65, 'phoneme': 70, 'character': 100}[unit],  # max lens from tcdtimit
            batch_normalisation=batch_normalisation,
            instance_normalisation=instance_normalisation,
            input_dense_layers=input_dense_layers,
            encoder_type=encoder_type,
            architecture=architecture,
            highway_encoder=highway_encoder,
            residual_encoder=residual_encoder,
            cell_type=cell_type,
            recurrent_l2_regularisation=None if optimiser == 'AdamW' else recurrent_l2_regularisation,
            weight_decay=weight_decay,
            encoder_units_per_layer=encoder_units_per_layer,
            decoder_units_per_layer=decoder_units_per_layer,
            bijective_state_copy=False,  # feature under testing
            enable_attention=enable_attention,
            attention_type=attention_type,
            use_dropout=use_dropout,
            dropout_probability=dropout_probability,
            embedding_size=embedding_size,
            sampling_probability_outputs=sampling_probability_outputs,
            decoding_algorithm=decoding_algorithm,
            beam_width=beam_width,
            use_ctc=False,
            optimiser=optimiser,
            loss_scaling=128 if precision == 'float16' else 1,
            learning_rate=learning_rate,
            lr_decay=lr_decay,
            loss_fun=loss_fun,
            clip_gradients=clip_gradients,
            max_gradient_norm=max_gradient_norm,
            num_gpus=num_gpus,
            write_attention_alignment=write_attention_alignment,
            dtype=tf.float16 if precision == 'float16' else tf.float32,
            profiling=profiling
        )

        self._hparams_audio = tf.contrib.training.HParams(
            frame_length_msec=25,  # 25 > 20
            frame_step_msec=10,
            sample_rate=22050,
            fft_length=1024,
            mel_lower_edge_hz=80,
            mel_upper_edge_hz=22050 / 2,  # 11025 > 7600
            num_mel_bins=30,  # 30 > 60 > 80
            num_mfccs=26,  # 26 > 13
        )

        self._hparams_video = tf.contrib.training.HParams(
            cnn_filters=cnn_filters,
            cnn_dense_units=cnn_dense_units,
        )

        self._create_graphs()
        self._create_models()
        self._create_sessions()
        self._initialize_sessions()

    def __del__(self):
        self._train_session.close()
        self._evaluate_session.close()
        self._predict_session.close()

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

            self._train_session.run([stream.iterator_initializer for stream in self._train_model.data
                                     if stream is not None])
            sum_loss = 0
            batches = 0

            start = time.time()

            try:
                while True:
                    out = self._train_session.run([self._train_model.model.train_op,
                                                   self._train_model.model.batch_loss,
                                                   ], **self.sess_opts)

                    if self._hparams.profiling is True:
                        self.profiler.add_step(batches, self.run_meta)

                        from tensorflow.python.profiler import option_builder

                        self.profiler.profile_name_scope(options=(option_builder.ProfileOptionBuilder
                                                                  .trainable_variables_parameter()))

                        opts = option_builder.ProfileOptionBuilder.time_and_memory()
                        self.profiler.profile_operations(options=opts)

                        opts = (option_builder.ProfileOptionBuilder(
                            option_builder.ProfileOptionBuilder.time_and_memory())
                                .with_step(batches)
                                .with_timeline_output('/tmp/timelines/').build())

                        self.profiler.profile_graph(options=opts)

                    sum_loss += out[1]
                    print('batch: {}'.format(batches))
                    batches += 1

            except tf.errors.OutOfRangeError:
                pass

            print('epoch time: {}'.format(time.time() - start))
            f.write('Average batch_loss as epoch {} is {}\n'.format(epoch, sum_loss / batches))
            f.flush()

            if epoch % 5 == 0:
                save_path = self._train_model.model.saver.save(
                    sess=self._train_session,
                    save_path=checkpoint_path,
                    global_step=epoch,
                )

                error_rate = self.evaluate(save_path, epoch)
                for (k, v) in error_rate.items():
                    f.write(k + ': {:.4f}% '.format(v * 100))
                f.write('\n')
                f.flush()

        f.close()

    def evaluate(self, checkpoint_path, epoch=None, alignments_outdir='./alignments/tmp/'):
        self._evaluate_model.model.saver.restore(
            sess=self._evaluate_session,
            save_path=checkpoint_path
        )
        self._evaluate_session.run([stream.iterator_initializer for stream in self._evaluate_model.data
                                    if stream is not None])
        predictions_dict = {}
        labels_dict = {}

        if self._hparams.video_processing is not None:
            data = self._evaluate_model.data[0]
        elif self._hparams.audio_processing is not None:
            data = self._evaluate_model.data[1]
        else:
            raise ValueError('At least one of A/V streams must be enabled')

        session_outputs = [self._evaluate_model.model._decoder.inference_predicted_ids,
                           data.labels,
                           data.inputs_filenames,
                           data.labels_filenames,]

        if self._write_attention_alignment is True:
            session_outputs.append(self._evaluate_model.model._decoder.attention_summary)
            if self._hparams.architecture == 'av_align':
                session_outputs.append(self._evaluate_model.model._audio_encoder.attention_summary)


        while True:

            try:
                out = self._evaluate_session.run(session_outputs)

                # debug time
                # assert (any(list(out[2] == out[3])))
                # assert (any(list(out[1] == out[3])))

                if self._write_attention_alignment is True:
                    imag_summ = tf.Summary()
                    imag_summ.ParseFromString(out[4])

                    if self._hparams.architecture == 'av_align':
                        cross_modal_summary = tf.Summary()
                        cross_modal_summary.ParseFromString(out[5])

                for idx in range(len(out[2])):  # could use batch_size here, but take care with the last smaller batch
                    predicted_ids = out[0][idx]
                    predicted_symbs = [self._unit_dict[sym] for sym in predicted_ids]

                    labels_ids = out[1][idx]
                    labels_symbs = [self._unit_dict[sym] for sym in labels_ids]

                    file = out[2][idx].decode('utf-8')

                    if self._write_attention_alignment is True:
                        fname = path.join(alignments_outdir, file + '.png')
                        makedirs(path.dirname(fname), exist_ok=True)
                        with tf.gfile.GFile(fname, mode='w') as img_f:
                            img_f.write(imag_summ.value[idx].image.encoded_image_string)

                        if self._hparams.architecture == 'av_align':
                            fname = path.join(alignments_outdir, file + '_av.png')
                            with tf.gfile.GFile(fname, mode='w') as img_f:
                                img_f.write(cross_modal_summary.value[idx].image.encoded_image_string)

                    predictions_dict[file] = predicted_symbs
                    labels_dict[file] = labels_symbs

            except tf.errors.OutOfRangeError:
                break

        uer = compute_wer(predictions_dict, labels_dict)
        error_rate = {self._unit: uer}
        if self._unit == 'character':
            wer = compute_wer(predictions_dict, labels_dict, split_words=True)
            error_rate['word'] = wer

        outdir = path.join('predictions', path.split(path.split(checkpoint_path)[0])[-1])
        makedirs(outdir, exist_ok=True)
        write_sequences_to_labelfile(predictions_dict,
                                     path.join(outdir, 'predicted_epoch_{}.mlf'.format(epoch)),
                                     labels_dict)

        return error_rate

    def _create_graphs(self):
        self._train_graph = tf.Graph()
        self._evaluate_graph = tf.Graph()
        self._predict_graph = tf.Graph()

    def _create_models(self):
        self._train_model = self._make_model(
            graph=self._train_graph,
            mode='train',
            batch_size=self._hparams.batch_size[0])
        self._evaluate_model = self._make_model(
            graph=self._evaluate_graph,
            mode='evaluate',
            batch_size=self._hparams.batch_size[1])

    def _create_sessions(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        self._train_session = tf.Session(graph=self._train_graph, config=config)
        self._evaluate_session = tf.Session(graph=self._evaluate_graph, config=config)
        self._predict_session = tf.Session(graph=self._predict_graph, config=config)

        if self._hparams.profiling is True:
            from tensorflow.profiler import Profiler
            self.profiler = Profiler(self._train_session.graph)
            self.run_meta = tf.RunMetadata()
            makedirs('/tmp/timelines/', exist_ok=True)
            self.sess_opts = {
                'options': tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                'run_metadata': self.run_meta
            }
        else:
            self.sess_opts = {}

    def _initialize_sessions(self):
        self._train_session.run(self._train_model.initializer)
        self._evaluate_session.run(self._evaluate_model.initializer)

    def _make_model(self, graph, mode, batch_size):
        with graph.as_default():

            video_data, audio_data = self._fetch_data(mode, batch_size)
            video_features, audio_features = self._preprocess_data(video_data, audio_data, mode, batch_size)

            model = Seq2SeqModel(
                data_sequences=(video_features, audio_features),
                mode=mode,
                hparams=self._hparams
            )

            initializer = tf.global_variables_initializer()

            # Returning the original data, not the processed features
            return Model(data=(video_data, audio_data),
                         model=model,
                         initializer=initializer,
                         batch_size=batch_size)

    def _parse_iterator(self, iterator):
        inputs = tf.cast(iterator.inputs, dtype=self._hparams.dtype)
        labels = tf.cast(iterator.labels, tf.int32, name='labels')
        inputs_length = tf.cast(iterator.inputs_len, tf.int32, name='inputs_len')
        labels_length = tf.cast(iterator.labels_len, tf.int32, name='labels_len')

        return Data(
            inputs=inputs,
            inputs_length=inputs_length,
            inputs_filenames=iterator.filename,
            labels=labels,
            labels_length=labels_length,
            labels_filenames=iterator.filename2,
            iterator_initializer=iterator.initializer)

    def _parse_multimodal_iterator(self, iterator):
        vid_inputs = tf.cast(iterator.inputs[0], dtype=self._hparams.dtype, name='vid_inputs')
        vid_inputs_length = tf.cast(iterator.inputs_len[0], tf.int32, name='vid_inputs_len')
        aud_inputs = tf.cast(iterator.inputs[1], dtype=self._hparams.dtype, name='aud_inputs')
        aud_inputs_length = tf.cast(iterator.inputs_len[1], tf.int32, name='aud_inputs_len')

        labels = tf.cast(iterator.labels, tf.int32, name='labels')
        labels_length = tf.cast(iterator.labels_len, tf.int32, name='labels_len')

        video_data = Data(
            inputs=vid_inputs,
            inputs_length=vid_inputs_length,
            inputs_filenames=iterator.filename[0],
            labels=labels,
            labels_length=labels_length,
            labels_filenames=iterator.filename2,
            iterator_initializer=iterator.initializer)

        audio_data = Data(
            inputs=aud_inputs,
            inputs_length=aud_inputs_length,
            inputs_filenames=iterator.filename[1],
            labels=labels,
            labels_length=labels_length,
            labels_filenames=iterator.filename2,
            iterator_initializer=iterator.initializer)

        return video_data, audio_data

    def _fetch_data(self, mode, batch_size):

        video_data = None
        audio_data = None

        if self._video_processing is not None and self._audio_processing is not None:

            iterator = make_iterator_from_two_records(
                video_record=self._video_train_record if mode == 'train' else self._video_test_record,
                audio_record=self._audio_train_record if mode == 'train' else self._audio_test_record,
                label_record=self._labels_train_record if mode == 'train' else self._labels_test_record,
                batch_size=batch_size,
                unit_dict=self._hparams.unit_dict,
                shuffle=True if mode == 'train' else False,
                reverse_input=False,
                bucket_width=45,  # video frame rate is 30 fps
            )
            video_data, audio_data = self._parse_multimodal_iterator(iterator)

        else:
            if self._video_processing is not None:
                video_iterator = make_iterator_from_one_record(
                    data_record=self._video_train_record if mode == 'train' else self._video_test_record,
                    label_record=self._labels_train_record if mode == 'train' else self._labels_test_record,
                    batch_size=batch_size,
                    unit_dict=self._hparams.unit_dict,
                    shuffle=True if mode == 'train' else False,
                    reverse_input=False,
                    bucket_width=45,  # video frame rate is 30 fps
                )

                video_data = self._parse_iterator(video_iterator)

            if self._audio_processing is not None:
                audio_iterator = make_iterator_from_one_record(
                    data_record=self._audio_train_record if mode == 'train' else self._audio_test_record,
                    label_record=self._labels_train_record if mode == 'train' else self._labels_test_record,
                    batch_size=batch_size,
                    shuffle=True if mode == 'train' else False,
                    unit_dict=self._hparams.unit_dict,
                    reverse_input=False,
                    bucket_width=45,  # audio feature rate is 30 mfcc/sec
                )

                audio_data = self._parse_iterator(audio_iterator)

        return video_data, audio_data

    def _preprocess_data(self, video_data, audio_data, mode, batch_size):

        if self._video_processing is not None:

            if 'cnn' in self._video_processing:

                visual_features = cnn_layers(
                    inputs=video_data.inputs,
                    cnn_type=self._video_processing,
                    is_training=(mode=='train'),
                    cnn_filters=self._hparams_video.cnn_filters,
                    cnn_dense_units=self._hparams_video.cnn_dense_units
                )

                # re-create video_data to update the `inputs` field
                video_data = Data(
                    inputs=visual_features,
                    inputs_length=video_data.inputs_length,
                    inputs_filenames=video_data.inputs_filenames,
                    labels=video_data.labels,
                    labels_length=video_data.labels_length,
                    labels_filenames=video_data.labels_filenames,
                    iterator_initializer=video_data.iterator_initializer
                )

            elif self._video_processing == 'features':
                pass
            else:
                raise Exception('unknown visual content')
        else:
            pass

        if self._audio_processing is not None:

            if self._audio_processing == 'wav':  # compute mfcc on the fly

                audio_features = process_audio(
                    audio_data.inputs,
                    hparams=self._hparams_audio,
                )

                # re-create audio_data to update the `inputs` field
                audio_data = Data(
                    inputs=audio_features,
                    inputs_length=audio_data.inputs_length,
                    inputs_filenames=audio_data.inputs_filenames,
                    labels=audio_data.labels,
                    labels_length=audio_data.labels_length,
                    labels_filenames=audio_data.labels_filenames,
                    iterator_initializer=audio_data.iterator_initializer
                )

            elif self._audio_processing == 'features':
                pass
            else:
                raise Exception('unknown audio content')
        else:
            pass

        return video_data, audio_data

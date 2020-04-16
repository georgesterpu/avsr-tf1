import tensorflow as tf
from tensorflow.contrib.mixed_precision import ExponentialUpdateLossScaleManager, LossScaleOptimizer
from .encoder import Seq2SeqEncoder, AttentiveEncoder
from .decoder_bimodal import Seq2SeqBimodalDecoder
from .decoder_unimodal import Seq2SeqUnimodalDecoder
from tensorflow.contrib import seq2seq


class Seq2SeqModel(object):
    def __init__(self,
                 data_sequences,
                 mode,
                 hparams):

        # member variables
        self._video_data = data_sequences[0]
        self._audio_data = data_sequences[1]

        self._mode = mode
        self._hparams = hparams

        self._make_encoders()
        self._make_decoder()

        if mode == 'train':
            self._init_optimiser()

        self._init_saver()

    def _make_encoders(self):
        if self._video_data is not None:
            with tf.variable_scope('video'):
                self._video_encoder = Seq2SeqEncoder(
                    data=self._video_data,
                    mode=self._mode,
                    hparams=self._hparams,
                    num_units_per_layer=self._hparams.encoder_units_per_layer[0],
                    dropout_probability=self._hparams.video_encoder_dropout_probability,
                    regress_aus=self._hparams.regress_aus,
                )
        else:
            self._video_encoder = None

        if self._audio_data is not None:
            with tf.variable_scope('audio'):

                if self._hparams.architecture in ('unimodal', 'bimodal',):
                    self._audio_encoder = Seq2SeqEncoder(
                        data=self._audio_data,
                        mode=self._mode,
                        hparams=self._hparams,
                        num_units_per_layer=self._hparams.encoder_units_per_layer[1],
                        dropout_probability=self._hparams.audio_encoder_dropout_probability,
                    )
                elif self._hparams.architecture == 'av_align':
                    self._audio_encoder = AttentiveEncoder(
                        data=self._audio_data,
                        mode=self._mode,
                        hparams=self._hparams,
                        num_units_per_layer=self._hparams.encoder_units_per_layer[1],
                        attended_memory=self._video_encoder.get_data().outputs,
                        attended_memory_length=self._video_data.inputs_length,
                        dropout_probability=self._hparams.audio_encoder_dropout_probability,
                    )
                else:
                    raise Exception('Unknown architecture')
        else:
            self._audio_encoder = None

    def _make_decoder(self):

        labels = None
        labels_length = None
        video_len = None
        audio_len = None

        if self._video_encoder is not None:
            video_output = self._video_encoder.get_data()
            labels = self._video_data.labels
            labels_length = self._video_data.labels_length
            video_len = self._video_data.inputs_length
        else:
            video_output = None

        if self._audio_encoder is not None:
            audio_output = self._audio_encoder.get_data()
            labels = self._audio_data.labels
            labels_length = self._audio_data.labels_length
            audio_len = self._audio_data.inputs_length
        else:
            audio_output = None

        if labels is None or labels_length is None:
            raise Exception('labels are None')

        if self._hparams.architecture in ('unimodal', 'av_align'):

            if audio_output is not None:
                output = audio_output
                features_len = audio_len
            else:
                output = video_output
                features_len = video_len

            self._decoder = Seq2SeqUnimodalDecoder(
                encoder_output=output,
                encoder_features_len=features_len,
                labels=labels,
                labels_length=labels_length,
                mode=self._mode,
                hparams=self._hparams
            )

        elif self._hparams.architecture == 'bimodal':
            self._decoder = Seq2SeqBimodalDecoder(
                video_output=video_output,
                audio_output=audio_output,
                video_features_len=video_len,
                audio_features_len=audio_len,
                labels=labels,
                labels_length=labels_length,
                mode=self._mode,
                hparams=self._hparams
            )
        else:
            raise Exception('Unknown architecture')

    def extract_results(self):

        pass

    def _init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=1)

    def _init_optimiser(self):
        r"""
                Computes the batch_loss function to be minimised
                """

        self._init_lr_decay()

        self._loss_weights = tf.sequence_mask(
            lengths=self._decoder._labels_len,
            dtype=self._hparams.dtype
        )

        if self._hparams.loss_fun is None:
            if self._hparams.label_smoothing <= 0.0:
                softmax_loss_fun = None
            else:
                print('Using the slower "softmax_cross_entropy" instead of "sparse_softmax_cross_entropy" '
                      'since label smoothing is nonzero')
                from .devel import smoothed_cross_entropy
                num_classes = tf.shape(self._decoder._logits)[2]
                softmax_loss_fun = smoothed_cross_entropy(num_classes, self._hparams.label_smoothing)
        elif self._hparams.loss_fun == 'focal_loss':
            from .devel import focal_loss
            softmax_loss_fun = focal_loss
        elif self._hparams.loss_fun == 'mc_loss':
            from .devel import mc_loss
            softmax_loss_fun = mc_loss
        else:
            raise ValueError('Unknown loss function {}'.format(self._hparams.loss_fun))

        self.batch_loss = seq2seq.sequence_loss(
            logits=self._decoder._logits,
            targets=self._decoder._labels,
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

        if self._hparams.regress_aus is True:
            loss_weight = self._hparams.kwargs.get('au_loss_weight', 10.0)
            self.batch_loss += loss_weight * self._video_encoder.au_loss

        if self._hparams.loss_scaling > 1:
            self.batch_loss *= self._hparams.loss_scaling

        if self._hparams.optimiser == 'Adam':
            optimiser = tf.train.AdamOptimizer(
                learning_rate=self.current_lr,
                epsilon=1e-8 if self._hparams.dtype == tf.float32 else 1e-4,
            )
        elif self._hparams.optimiser == 'Nadam':
            from tensorflow.contrib.opt import NadamOptimizer
            optimiser = NadamOptimizer(
                learning_rate=self.current_lr,
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
        gradients = tf.gradients(self.batch_loss, variables)  # not compatible with Nvidia AMP (fp16)
        # gradients = optimiser.compute_gradients(self.batch_loss, variables)

        summaries= []
        for grad, variable in zip(gradients, variables):
            if isinstance(grad, tf.IndexedSlices):
                value = grad.values
            else:
                value = grad
            summary = tf.summary.histogram("%s-grad" % variable.name, value)
            summaries.append(summary)

        if self._hparams.dtype == tf.float16:
            #ripped from https://github.com/joeyearsley/efficient_densenet_tensorflow/blob/master/train.py
            # Choose a loss scale manager which decides how to pick the right loss scale
            # throughout the training process.
            loss_scale_manager = ExponentialUpdateLossScaleManager(128, 100)
            # Wraps the original optimizer in a LossScaleOptimizer.
            optimizer = LossScaleOptimizer(optimiser, loss_scale_manager)

        if self._hparams.loss_scaling > 1:
            gradients = [tf.div(grad, self._hparams.loss_scaling) for grad in gradients]

        if self._hparams.clip_gradients is True:
            gradients, self.global_norm = tf.clip_by_global_norm(gradients, self._hparams.max_gradient_norm)

        if self._hparams.batch_normalisation is True:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimiser.apply_gradients(
                    grads_and_vars=zip(gradients, variables),
                    global_step=tf.train.get_global_step())
        else:
            self.train_op = optimiser.apply_gradients(
                grads_and_vars=zip(gradients, variables),
                global_step=tf.train.get_global_step())

    def _init_lr_decay(self):

        self.global_step = tf.train.get_global_step()

        if self._hparams.lr_decay is None:
            self.current_lr = self._hparams.learning_rate

        elif self._hparams.lr_decay[0] == 'cosine_restarts':
            self.current_lr = tf.train.cosine_decay_restarts(
                learning_rate=self._hparams.learning_rate,
                global_step=tf.train.get_global_step(),
                first_decay_steps=self._hparams.lr_decay[1])
        else:
            print('learning rate policy not implemented, falling back to constant learning rate')
            self.current_lr = self._hparams.learning_rate

        steps = self._hparams.kwargs.get('warmup_steps', 750)
        if steps:
            global_step = tf.to_float(tf.train.get_global_step())
            warmup_steps = tf.to_float(steps)
            scale = tf.minimum(1.0, (global_step + 1) / warmup_steps)
            self.current_lr *= scale


def _get_trainable_vars(cell_type):
    r"""
    Returns the list of trainable variables associated with the recurrent layers
    """
    cell_type = cell_type.split('_')[0]
    vars_ = [var for var in tf.trainable_variables() if cell_type + '_' in var.name
             and 'bias' not in var.name]
    return vars_

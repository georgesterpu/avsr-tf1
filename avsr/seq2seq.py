import tensorflow as tf
from .encoder import Seq2SeqEncoder
from .decoder_bimodal import Seq2SeqBimodalDecoder
from .decoder_unimodal import Seq2SeqUnimodalDecoder
# from .avsr import Data


class Seq2SeqModel(object):
    def __init__(self,
                 data_sequences,
                 mode,
                 hparams):

        # member variables
        self._video_data = data_sequences[0]
        self._audio_data = data_sequences[1]
        
        #in some cases the dict is wrapped in a tuple
        try:
            self.attention_type_enc = hparams.attention_type_enc[0]
        except:
            self.attention_type_enc = hparams.attention_type_enc
            
        try:
            self.output_attention_enc = hparams.output_attention_enc[0]
        except:
            self.output_attention_enc = hparams.output_attention_enc

        self._mode = mode
        self._hparams = hparams

        self._make_encoders()
        self._make_decoder()
        self._init_saver()

    def _make_encoders(self):

        #need to make a variable so we can pass a reference
        #while video output doesn't exist yet
        #need to fake memory
        att_memory_audio = None
        encoder_output_size = self._hparams.encoder_units_per_layer[-1]
        #only last dim counts, but tf.zeros needs full spec
        fake_mem = tf.zeros([1, 1, encoder_output_size])
        #underspecified output shape as dynamic_rnn produces
        encoder_output_shape = tf.TensorShape([None, None, encoder_output_size])

        #att_memory_audio = tf.Variable(fake_mem)
        #att_memory_audio.set_shape(encoder_output_shape)
            
        #print('att_memory_audio shape ', att_memory_audio.shape)
        if self._video_data is not None:

            if self._audio_data is not None:
                if (self._hparams.enable_attention_enc is True and
                    (len(self.attention_type_enc['video']['o2s']) > 0)):
                        #produce a memory with acceptable size for initing real encoder
                        att_memory_audio = tf.Variable(fake_mem, name="att_memory_audio")
                        att_memory_audio.set_shape(encoder_output_shape)

            with tf.variable_scope('video'):
                self._video_encoder = Seq2SeqEncoder(
                    data=self._video_data,
                    mode=self._mode,
                    hparams=self._hparams,
                    gpu_id=0 % self._hparams.num_gpus,
                    attention_type=self.attention_type_enc['video'],
                    output_attention=self.output_attention_enc['video'],
                    att_memory=att_memory_audio, #fake so far, need rewiring
                    att_memory_len=self._audio_data.inputs_length
                )
        else:
            self._video_encoder = None
        #print("made video encoder ")
                
        if self._audio_data is not None:
            with tf.variable_scope('audio'):
                self._audio_encoder = Seq2SeqEncoder(
                    data=self._audio_data,
                    mode=self._mode,
                    hparams=self._hparams,
                    gpu_id=1 % self._hparams.num_gpus,
                    attention_type=self.attention_type_enc['audio'],
                    output_attention=self.output_attention_enc['audio'],
                    att_memory=self._video_encoder.get_data().outputs,
                    att_memory_len=self._video_data.inputs_length
                )
                
            if self._video_data is not None:
                try:
                    #disable shape checking output as is [?,?, num_units[-1]]
                    #and we inited with [1,1, num_units[-1]]
                    att_memory_audio.assign(self._audio_encoder.get_data().outputs, validate_shape=False)
                    #set expected shape
                    att_memory_audio.set_shape(self._audio_encoder.get_data().outputs.get_shape())
                except:
                    att_memory_audio = self._audio_encoder.get_data().outputs
                
            att_memory_audio = self._audio_encoder.get_data().outputs

        else:
            self._audio_encoder = None
        #print("made audio encoder ")
        #print('att_memory_audio shape ', att_memory_audio.shape)

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

        self._decoder = Seq2SeqBimodalDecoder(
            video_output=video_output,
            audio_output=audio_output,
            video_features_len=video_len,
            audio_features_len=audio_len,
            labels=labels,
            labels_length=labels_length,
            mode=self._mode,
            hparams=self._hparams,
            gpu_id=1 % self._hparams.num_gpus
        )
        # self._decoder = Seq2SeqUnimodalDecoder(
        #     encoder_output=video_output,
        #     encoder_features_len=video_len,
        #     labels=labels,
        #     labels_length=labels_length,
        #     mode=self._mode,
        #     hparams=self._hparams
        # )

        self.train_op = self._decoder.train_op if self._mode == 'train' else None
        self.batch_loss = self._decoder.batch_loss if self._mode == 'train' else None

    def extract_results(self):

        pass

    def _init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=300)

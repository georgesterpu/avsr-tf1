import tensorflow as tf
from tensorflow.contrib import seq2seq
import collections
from .cells import build_rnn_layers
from .attention import create_attention_mechanism, create_attention_mechanism
from .utils import flatten
from tensorflow.python.layers.core import Dense


class EncoderData(collections.namedtuple("EncoderData", ("outputs", "final_state"))):
    pass


class Seq2SeqEncoder(object):

    def __init__(self,
                 data,
                 mode,
                 hparams,
                 gpu_id,
                 attention_type,
                 output_attention,
                 att_memory,
                 att_memory_len=None,                
                 ):

        self._data = data
        self._mode = mode
        self._hparams = hparams
        self._gpu_id = gpu_id
        
        self._output_attention = output_attention
        self._attention_type = attention_type
        self._att_memory = att_memory
        self._att_memory_len = att_memory_len
        
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
            
            #encoder outputs are needed for self-attention
            #they need to be passed when attention mechanisms are created
            #however they only exist after the encoder is created
            #but the encoder needs the attention mechanisms to build
            
            #outputs, state = tf.nn.dynamic_rnn()
            #outputs: The RNN output Tensor shaped [batch_size, max_time, cell.output_size]
            #when tf.nn.dynamic_rnn() is invoked, outputs.shape == [?,?, cell.output_size]
            #as these depend on the actual batches returned from the data feed
            #the AttentionMechanism.__init__() check that memory.shape[2:].fully_defined()
            #i.e. the last dimension must be known
            #we cannot build a Variable from outputs as the dimensions must be fully
            #known
            #when we use Variable(...,validate_shape=False) the shape is set to [None]
            #and the check in AttentionMechanism fails

            encoder_inputs = self._maybe_add_dense_layers()
            self._encoder_outputs = None
            # encoder_inputs = a_resnet(encoder_inputs, self._mode == 'train')
                    

            if self._hparams.encoder_type == 'unidirectional':
                self._encoder_cells = build_rnn_layers(
                    cell_type=self._hparams.cell_type,
                    num_units_per_layer=self._hparams.encoder_units_per_layer,
                    use_dropout=self._hparams.use_dropout,
                    dropout_probability=self._hparams.dropout_probability,
                    mode=self._mode,
                    base_gpu=self._gpu_id)
                
                #only last dim is relevant, but others are needed for tf.zeros
                fake_mem = tf.zeros([1, 1, self._encoder_cells.output_size])
                encoder_output_shape = tf.TensorShape([None, None, self._encoder_cells.output_size])
                if (self._hparams.enable_attention_enc is True and
                    (len(self._attention_type['o2s']) + len(self._attention_type['s2s']) > 0)):
                    
                    if len(self._attention_type['s2s']) > 0:
                        #this is used in self._create_attention_mechanisms()
                        self._encoder_outputs = tf.Variable(fake_mem, validate_shape=False, name="enc_out_fake")
                        #reset shape to underspecified 
                        self._encoder_outputs.set_shape(encoder_output_shape)
                    else:
                        self._encoder_outputs = None
                    
                    attention_mechanisms, layer_sizes = self._create_attention_mechanisms()
                    
                    if (len(attention_mechanisms) > 0):
                        batch_size = self._hparams.batch_size[0 if self._mode == 'train' else 1]
                        attention_cells = seq2seq.AttentionWrapper(
                            cell=self._encoder_cells,
                            attention_mechanism=attention_mechanisms,
                            attention_layer_size=layer_sizes,
                            initial_cell_state=self._encoder_cells.zero_state(batch_size, dtype=self._hparams.dtype),
                            alignment_history=False,
                            output_attention=self._output_attention,
                        )
                        #batch_size, _ = tf.unstack(tf.shape(self._labels))
            
                        #attn_zero = attention_cells.zero_state(
                        #   dtype=self._hparams.dtype, batch_size=batch_size
                        #)
                        #initial_state = attn_zero.clone(
                        #    cell_state=self._decoder_initial_state
                        #)
            
                        self._encoder_cells = attention_cells
                
                sequence_length = self._inputs_len
                #sequence_length = tf.Print(self._inputs_len, [self._inputs_len, tf.shape(self._inputs_len)], "sl::"+self._inputs_len.name, summarize=10)
                #encoder_inputs = tf.Print(encoder_inputs, [encoder_inputs, tf.shape(encoder_inputs)], "ei::"+encoder_inputs.name) 
                encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                cell=self._encoder_cells,
                inputs=encoder_inputs,
                sequence_length=sequence_length,
                parallel_iterations=self._hparams.batch_size[0 if self._mode == 'train' else 1],
                swap_memory=self._hparams.swap_memory,
                dtype=self._hparams.dtype,
                scope=scope,
                )
                self._encoder_final_state = encoder_state
                
                
                #print('encoder_outputs shape ', encoder_outputs.shape)
                try:
                    #if self-attention: re-wire to real encoder 
                    tf.assign(self._encoder_outputs, encoder_outputs)
                    self._encoder_outputs.set_shape(encoder_output_shape)
                except:
                    self._encoder_outputs = encoder_outputs
                #print('self._encoder_outputs shape ', self._encoder_outputs.shape)
                #self._encoder_outputs=tf.Print(self._encoder_outputs,
                #                                [self._encoder_outputs,
                #                                 tf.shape(self._encoder_outputs)],
                #                               self._encoder_outputs.name)

            elif self._hparams.encoder_type == 'bidirectional':

                # num_bi_layers = max(1, int(len(self._hparams.encoder_units_per_layer)/2))
                self._fw_cells = build_rnn_layers(
                    cell_type=self._hparams.cell_type,
                    num_units_per_layer=self._hparams.encoder_units_per_layer,
                    use_dropout=self._hparams.use_dropout,
                    dropout_probability=self._hparams.dropout_probability,
                    mode=self._mode,
                    base_gpu=self._gpu_id
                )  # note that we half the num_layers

                self._bw_cells = build_rnn_layers(
                    cell_type=self._hparams.cell_type,
                    num_units_per_layer=self._hparams.encoder_units_per_layer,
                    use_dropout=self._hparams.use_dropout,
                    dropout_probability=self._hparams.dropout_probability,
                    mode=self._mode,
                    base_gpu=self._gpu_id
                )

                bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=self._fw_cells,
                    cell_bw=self._bw_cells,
                    inputs=encoder_inputs,
                    sequence_length=self._inputs_len,
                    dtype=self._hparams.dtype,
                    parallel_iterations=self._hparams.batch_size[0 if self._mode == 'train' else 1],
                    swap_memory=self._hparams.swap_memory,
                    scope=scope
                )

                self._encoder_outputs = tf.concat(bi_outputs, -1)
                # if num_bi_layers != 1:
                #     self._encoder_final_state = bi_state
                # else:
                encoder_state = []
                for layer in range(len(bi_state[0])):
                    encoder_state.append(bi_state[0][layer])  # fw
                    encoder_state.append(bi_state[1][layer])  # bw
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
        
    def _create_attention_mechanisms(self, num_units=0):

        self._attention_output= []
        mechanisms = []
        layer_sizes = []
        
        
        num_units = (num_units if num_units > 0 else 
                    self._hparams.encoder_units_per_layer[-1])

        if self._att_memory is not None:

            for attention_type_ in flatten(self._attention_type['o2s']):

                #XXX: is -1 the right assumption? it's the last added layer
                #i.e. the topmost
                attention_video, attention_output = create_attention_mechanism(
                    num_units=num_units,
                    memory=self._att_memory,
                    memory_sequence_length=self._att_memory_len,
                    attention_type=attention_type_
                )
                mechanisms.append(attention_video)
                layer_sizes.append(num_units)
            
                self._attention_output.append(attention_output) 
            
        if self._encoder_outputs is not None:
            
            for attention_type_ in flatten(self._attention_type['s2s']):
    
                #XXX: is -1 the right assumption? it's the last added layer
                #i.e. the topmost
                attention_video_self, attention_output = create_attention_mechanism(
                    num_units=num_units,
                    memory=self._encoder_outputs,
                    memory_sequence_length=self._inputs_len,
                    attention_type=attention_type_
                )
                mechanisms.append(attention_video_self)
                layer_sizes.append(num_units)
            
                self._attention_output.append(attention_output)


        return mechanisms, layer_sizes
    

    

import tensorflow as tf
from tensorflow.contrib import seq2seq
from .cells import build_rnn_layers
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops import array_ops
from tensorflow.contrib.rnn import LSTMStateTuple

def create_attention_mechanism(attention_type,
                                num_units,
                                memory,
                                memory_sequence_length,
                                mode='train'):
    memory = tf.Print(memory, [memory, tf.shape(memory)], "attention memory dimensions ")
    memory_sequence_length = tf.Print(memory_sequence_length, 
                                      [memory_sequence_length, tf.shape(memory_sequence_length)], 
                                      "attention memory sequence_length ")
    if attention_type == 'bahdanau':
        attention_mechanism = seq2seq.BahdanauAttention(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            normalize=False
        )
        output_attention = False
    elif attention_type == 'normed_bahdanau':
        attention_mechanism = seq2seq.BahdanauAttention(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            normalize=True
        )
        output_attention = False
    elif attention_type == 'normed_monotonic_bahdanau':
        attention_mechanism = seq2seq.BahdanauMonotonicAttention(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            normalize=True,
            score_bias_init=-2.0,
            sigmoid_noise=1.0 if mode == 'train' else 0.0,
            mode='hard' if mode != 'train' else 'parallel'
        )
        output_attention = False
    elif attention_type == 'luong':
        attention_mechanism = seq2seq.LuongAttention(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_sequence_length
        )
        output_attention = True
    elif attention_type == 'scaled_luong':
        attention_mechanism = seq2seq.LuongAttention(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            scale=True,
        )
        output_attention = True
    elif attention_type == 'scaled_monotonic_luong':
        attention_mechanism = seq2seq.LuongMonotonicAttention(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            scale=True,
            score_bias_init=-2.0,
            sigmoid_noise=1.0 if mode == 'train' else 0.0,
            mode='hard' if mode != 'train' else 'parallel'
        )
        output_attention = True
    else:
        raise Exception('unknown attention mechanism ',attention_type)

    return attention_mechanism, output_attention

def create_attention_alignments_summary(states):
    attention_alignment = states.alignment_history.stack()

    attention_images = tf.expand_dims(tf.transpose(attention_alignment, [1, 2, 0]), -1)

    # attention_images_scaled = tf.image.resize_images(1-attention_images, (256,128))
    attention_images_scaled = 1 - attention_images

    attention_summary = tf.summary.image("attention_images", attention_images_scaled,
                                         max_outputs=self._hparams.batch_size[1])

    return attention_summary
import tensorflow as tf
from tensorflow.contrib import seq2seq


def create_attention_mechanism(
        attention_type,
        num_units,
        memory,
        memory_sequence_length,
        mode,
        dtype):
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
            dtype=dtype
        )
        output_attention = False
    elif attention_type == 'normed_bahdanau':
        attention_mechanism = seq2seq.BahdanauAttention(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            normalize=True,
            dtype=dtype,
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
            mode='hard' if mode != 'train' else 'parallel',
            dtype=dtype,
        )
        output_attention = False

    elif attention_type == 'luong':
        attention_mechanism = seq2seq.LuongAttention(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            dtype=dtype,
        )
        output_attention = True
    elif attention_type == 'scaled_luong':
        attention_mechanism = seq2seq.LuongAttention(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            scale=True,
            dtype=dtype,
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
            mode='hard' if mode != 'train' else 'parallel',
            dtype=dtype,
        )
        output_attention = True
    else:
        raise Exception('unknown attention mechanism')

    return attention_mechanism, output_attention


def create_attention_mechanisms(num_units, attention_types, mode, dtype, beam_search=False, beam_width=None, memory=None, memory_len=None, fusion_type=None):
    r"""
    Creates a list of attention mechanisms (e.g. seq2seq.BahdanauAttention)
    and also a list of ints holding the attention projection layer size
    Args:
        beam_search: `bool`, whether the beam-search decoding algorithm is used or not
    """
    mechanisms = []
    output_attention = None

    if beam_search is True:
        memory = seq2seq.tile_batch(
            memory, multiplier=beam_width)

        memory_len = seq2seq.tile_batch(
            memory_len, multiplier=beam_width)

    for attention_type in attention_types:
        attention, output_attention = create_attention_mechanism(
            num_units=num_units,  # has to match decoder's state(query) size
            memory=memory,
            memory_sequence_length=memory_len,
            attention_type=attention_type,
            mode=mode,
            dtype=dtype,
        )
        mechanisms.append(attention)

    N = len(attention_types)
    if fusion_type == 'deep_fusion':
        attention_layer_sizes = None
        attention_layers = [AttentionLayers(units=num_units, dtype=dtype) for _ in range(N)]
    elif fusion_type == 'linear_fusion':
        attention_layer_sizes = [num_units, ] * N
        attention_layers = None
    else:
        raise Exception('Unknown fusion type')

    return mechanisms, attention_layers, attention_layer_sizes, output_attention


def add_attention(
        cells,
        attention_types,
        num_units,
        memory,
        memory_len,
        mode,
        batch_size,
        dtype,
        beam_search=False,
        beam_width=None,
        initial_state=None,
        write_attention_alignment=False,
        fusion_type='linear_fusion',
):
    r"""
    Wraps the decoder_cells with an AttentionWrapper
    Args:
        cells: instances of `RNNCell`
        beam_search: `bool` flag for beam search decoders
        batch_size: `Tensor` containing the batch size. Necessary to the initialisation of the initial state

    Returns:
        attention_cells: the Attention wrapped decoder cells
        initial_state: a proper initial state to be used with the returned cells
    """
    attention_mechanisms, attention_layers, attention_layer_sizes, output_attention = create_attention_mechanisms(
        beam_search=beam_search,
        beam_width=beam_width,
        memory=memory,
        memory_len=memory_len,
        num_units=num_units,
        attention_types=attention_types,
        fusion_type=fusion_type,
        mode=mode,
        dtype=dtype)

    if beam_search is True:
        initial_state= seq2seq.tile_batch(
            initial_state, multiplier=beam_width)

    attention_cells = seq2seq.AttentionWrapper(
        cell=cells,
        attention_mechanism=attention_mechanisms,
        attention_layer_size=attention_layer_sizes,
        # initial_cell_state=decoder_initial_state,
        alignment_history=write_attention_alignment,
        output_attention=output_attention,
        attention_layer=attention_layers,
    )

    attn_zero = attention_cells.zero_state(
        dtype=dtype,
        batch_size=batch_size * beam_width if beam_search is True else batch_size)

    if initial_state is not None:
        initial_state = attn_zero.clone(
            cell_state=initial_state)

    return attention_cells, initial_state


class AttentionLayers(object):
    def __init__(self, units, dtype):
        self.layer1 = tf.layers.Dense(
            units=units,
            name="attention_layer1",
            use_bias=False,
            dtype=dtype,
            activation=tf.nn.tanh,
        )

        self.layer2 = tf.layers.Dense(
            units=units,
            name="attention_layer2",
            use_bias=False,
            dtype=dtype,
            activation=tf.nn.tanh,
        )

    def compute_output_shape(self, input_shape):
        return self.layer2.compute_output_shape(input_shape)

    def __call__(self, vector):
        r"""
        Example for M4 (except sigmoid activation)
        a = vector[:, 0:256]
        v = vector[:, 256:]
        wa = self.layer1(a)
        wv = self.layer2(v)
        output = a*wa + v*wv
        """

        #  This is M1
        hidden = self.layer1(vector)
        output = self.layer2(hidden)
        return output

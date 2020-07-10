import tensorflow as tf
import collections
from .audio import process_audio
import numpy as np
from os import path


class BatchedData(collections.namedtuple("BatchedData",
                                         ("iterator_initializer",
                                          "inputs",
                                          "inputs_length",
                                          "inputs_filenames",
                                          "labels",
                                          "labels_length",
                                          "labels_filenames",
                                          "payload",
                                          ))):
    pass


def _parse_input_function(example, input_shape, content_type):

    if content_type['stream'] == 'feature':

        context_features = {
            "input_length": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
            "input_size": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
            "filename": tf.io.FixedLenFeature(shape=[], dtype=tf.string)
        }
    elif content_type['stream'] == 'video':
        context_features = {
            "input_length": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
            "width": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
            "height": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
            "channels": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
            "filename": tf.io.FixedLenFeature(shape=[], dtype=tf.string)
        }
    else:
        raise Exception('unknown content type')

    sequence_features = {
        "inputs": tf.io.FixedLenSequenceFeature(shape=input_shape, dtype=tf.float32)
    }

    if content_type.get('aus', False):
        sequence_features['aus'] = tf.io.FixedLenSequenceFeature(shape=[2], dtype=tf.float32)

    context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
        serialized=example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    sequence_output = [sequence_parsed["inputs"], ]

    if content_type.get('aus', False):
        sequence_output.append(sequence_parsed["aus"])

    context_output = [context_parsed["input_length"], context_parsed["filename"]]

    return sequence_output + context_output


def _parse_labels_function(example, unit_dict):
    context_features = {
        "unit": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "labels_length": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        "filename": tf.io.FixedLenFeature(shape=[], dtype=tf.string)
    }
    sequence_features = {
        "labels": tf.io.FixedLenSequenceFeature(shape=[], dtype=tf.int64)
    }

    context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
        serialized=example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    ivdict = {v: k for k, v in unit_dict.items()}
    labels = tf.concat([sequence_parsed["labels"], [ivdict['EOS']]], axis=0)

    labels_length = context_parsed["labels_length"] + 1

    return labels, labels_length, context_parsed["filename"]


def make_iterator_from_one_record(data_record, label_record, unit_dict, batch_size, shuffle=False, reverse_input=False, bucket_width=-1, num_cores=4, max_sentence_length=None):

    input_shape, content_type = _get_input_shape_from_record(data_record)
    # unit = _get_unit_from_record(label_record)
    has_aus = content_type.get('aus', False)

    dataset1 = tf.data.TFRecordDataset(data_record, num_parallel_reads=num_cores)
    dataset1 = dataset1.map(lambda proto: _parse_input_function(proto, input_shape, content_type), num_parallel_calls=num_cores)

    dataset2 = tf.data.TFRecordDataset(label_record, num_parallel_reads=num_cores)
    dataset2 = dataset2.map(lambda proto: _parse_labels_function(proto, unit_dict), num_parallel_calls=num_cores)

    dataset = tf.data.Dataset.zip((dataset1, dataset2))

    if max_sentence_length is not None:
        dataset = dataset.filter(lambda audio, labels: labels[1] < max_sentence_length)

    if shuffle is True:
        dataset = dataset.shuffle(buffer_size=5000, reshuffle_each_iteration=True)

    if reverse_input is True:
        dataset = dataset.map(
            lambda d1a, d1b, d1c, d2a, d2b, d2c: (tf.reverse(d1a, axis=[0]), d1b, d1c, d2a, d2b, d2c),
        )

    def batching_fun(x):
        if has_aus:
            data_shape = (tf.TensorShape([None] + input_shape), tf.TensorShape([None, 2]), tf.TensorShape([]),
                           tf.TensorShape([]))
        else:
            data_shape = (tf.TensorShape([None] + input_shape), tf.TensorShape([]), tf.TensorShape([]))

        labels_shape = (tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([]))
        return x.padded_batch(
            batch_size=batch_size,
            padded_shapes=(
                data_shape,
                labels_shape
            ), drop_remainder=False,
        )

    if bucket_width == -1:
        dataset = batching_fun(dataset)
    else:

        def key_func(arg1, arg2):
            # inputs_len = tf.shape(arg1[0])[0]
            inputs_len = arg1[-2]
            bucket_id = inputs_len // bucket_width
            return bucket_id

        def reduce_func(unused_key, windowed_dataset):
            return batching_fun(windowed_dataset)

        dataset = tf.data.Dataset.apply(dataset, tf.data.experimental.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    iterator = dataset.make_initializable_iterator()

    payload = {}
    if has_aus:
        (inputs, aus_inputs, inputs_len, fname1), (labels, labels_len, fname2) = iterator.get_next()
        payload['aus'] = aus_inputs
    else:
        (inputs, inputs_len, fname1), (labels, labels_len, fname2) = iterator.get_next()

    return BatchedData(
        iterator_initializer=iterator.initializer,
        inputs_filenames=fname1,
        labels_filenames=fname2,
        inputs=inputs,
        inputs_length=inputs_len,
        labels=labels,
        labels_length=labels_len,
        payload=payload,
    )


def make_iterator_from_two_records(video_record, audio_record, label_record, batch_size, unit_dict, shuffle=False, reverse_input=False, bucket_width=-1, num_cores=4):
    # TODO: this function needs a generalisation to lists of data records

    # unit = _get_unit_from_record(label_record)

    vid_input_shape, vid_content_type = _get_input_shape_from_record(video_record)
    vid_dataset = tf.data.TFRecordDataset(video_record, num_parallel_reads=num_cores)
    vid_dataset = vid_dataset.map(lambda proto: _parse_input_function(proto, vid_input_shape, vid_content_type), num_parallel_calls=num_cores)

    # has_aus = vid_content_type['aus'] if 'aus' in vid_content_type.keys() else None
    has_aus = vid_content_type.get('aus', False)

    aud_input_shape, aud_content_type = _get_input_shape_from_record(audio_record)
    aud_dataset = tf.data.TFRecordDataset(audio_record, num_parallel_reads=num_cores)
    aud_dataset = aud_dataset.map(lambda proto: _parse_input_function(proto, aud_input_shape, aud_content_type), num_parallel_calls=num_cores)

    # Why did I zip these two before zipping with the labels ?
    dataset1 = tf.data.Dataset.zip((vid_dataset, aud_dataset))

    dataset2 = tf.data.TFRecordDataset(label_record, num_parallel_reads=num_cores)
    dataset2 = dataset2.map(lambda proto: _parse_labels_function(proto, unit_dict), num_parallel_calls=num_cores)

    dataset = tf.data.Dataset.zip((dataset1, dataset2))

    if shuffle is True:
        dataset = dataset.shuffle(buffer_size=5000, reshuffle_each_iteration=True)

    # TODO (cba, two stacked lists of inputs)
    # if reverse_input is True:
    #     dataset = dataset.map(
    #         lambda d1a, d1b, d1c, d2a, d2b, d2c: (tf.reverse(d1a, axis=[0]), d1b, d1c, d2a, d2b, d2c),
    #     )

    def batching_fun(x, has_aus):

        if has_aus:
            video_shape = (
                tf.TensorShape([None] + vid_input_shape), tf.TensorShape([None, 2]), tf.TensorShape([]), tf.TensorShape([]),)  # shut up, PEP 8
        else:
            video_shape = (
                tf.TensorShape([None] + vid_input_shape), tf.TensorShape([]), tf.TensorShape([]),)

        audio_shape = (tf.TensorShape([None] + aud_input_shape), tf.TensorShape([]), tf.TensorShape([]),)
        labels_shape = (tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([]))

        return x.padded_batch(
            batch_size=batch_size,
            padded_shapes=(
                (video_shape, audio_shape), labels_shape
            )
        )

    if bucket_width == -1:
        dataset = batching_fun(dataset, has_aus)
    else:

        def key_func(arg1, arg2):
            inputs_len = tf.shape(arg1[0][0])[0]
            bucket_id = inputs_len // bucket_width
            return tf.cast(bucket_id, dtype=tf.int64)

        def reduce_func(unused_key, windowed_dataset):
            return batching_fun(windowed_dataset, has_aus)

        dataset = tf.data.Dataset.apply(dataset, tf.data.experimental.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    iterator = dataset.make_initializable_iterator()

    payload = {}
    if has_aus is True:
        ((vid_inputs, aus_inputs, vid_inputs_len, vid_fname1),
         (aud_inputs, aud_inputs_len, aud_fname1)), (labels, labels_len, labels_fname2) = iterator.get_next()

        payload['aus'] = aus_inputs
    else:
        # aus_inputs = None
        ((vid_inputs, vid_inputs_len, vid_fname1),
         (aud_inputs, aud_inputs_len, aud_fname1)), (labels, labels_len, labels_fname2) = iterator.get_next()

    return BatchedData(
        iterator_initializer=iterator.initializer,
        inputs_filenames=(vid_fname1, aud_fname1),
        labels_filenames=labels_fname2,
        inputs=(vid_inputs, aud_inputs),
        payload=payload,
        inputs_length=(vid_inputs_len, aud_inputs_len),
        labels=labels,
        labels_length=labels_len
    )


def make_iterator_from_label_record(label_record, batch_size, unit_dict, shuffle=False, reverse_input=False, bucket_width=-1, num_cores=4):
    dataset = tf.data.TFRecordDataset(label_record)
    dataset = dataset.map(lambda proto: _parse_labels_function(proto, unit_dict), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if shuffle is True:
        dataset = dataset.shuffle(buffer_size=45000, reshuffle_each_iteration=True)

    def batching_fun(x):

        labels_shape = (tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([]))

        return x.padded_batch(
            batch_size=batch_size,
            padded_shapes=(labels_shape)
        )

    if bucket_width == -1:
        dataset = batching_fun(dataset)
    else:

        def key_func(labels, labels_len, fname):
            # labels_len = tf.shape(labels)[0]
            bucket_id = labels_len // bucket_width
            return tf.cast(bucket_id, dtype=tf.int64)

        def reduce_func(unused_key, windowed_dataset):
            return batching_fun(windowed_dataset)

        dataset = tf.data.Dataset.apply(dataset, tf.data.experimental.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    iterator = dataset.make_initializable_iterator()

    labels, labels_len, labels_fname = iterator.get_next()

    return BatchedData(
        iterator_initializer=iterator.initializer,
        inputs_filenames=None,
        labels_filenames=labels_fname,
        inputs=None,
        payload=None,
        inputs_length=None,
        labels=labels,
        labels_length=labels_len
    )



def _get_input_shape_from_record(record):
    record_iterator = tf.python_io.tf_record_iterator(path=record)

    example = tf.train.SequenceExample()
    example.ParseFromString(record_iterator.__next__())

    # TODO(gsterpu) infer field names from record, avoid try/except
    content_type = {}

    try:  # one-dimensional input
        input_shape = [example.context.feature["input_size"].int64_list.value[0]]
        content_type['stream'] = 'feature'

    except:  # two dimensional input
        width = example.context.feature["width"].int64_list.value[0]
        height = example.context.feature["height"].int64_list.value[0]

        try:
            channels = example.context.feature["channels"].int64_list.value[0]
        except:
            channels = 1

        input_shape = [width, height, channels]
        content_type['stream'] = 'video'

    try:  # Action Units ?
        _ = example.feature_lists.feature_list['aus'].feature[0].float_list.value[0]
        content_type['aus'] = True
    except:
        pass

    return input_shape, content_type


def _get_unit_from_record(record):
    record_iterator = tf.python_io.tf_record_iterator(path=record)

    example = tf.train.SequenceExample()
    example.ParseFromString(record_iterator.__next__())
    unit = example.context.feature["unit"].bytes_list.value[0]
    return unit.decode('utf-8')


def create_unit_dict(unit_file):

    unit_dict = {'MASK': 0, 'END': -1}

    with open(unit_file, 'r') as f:
        unit_list = f.read().splitlines()

    idx = 0
    for idx, subunit in enumerate(unit_list):
        unit_dict[subunit] = idx + 1

    unit_dict['EOS'] = idx + 2
    unit_dict['GO'] = idx + 3

    ivdict = {v: k for k, v in unit_dict.items()}

    return ivdict


# Possible feature: decode files on the fly
#
# def parse_files_function(example):
#     from tensorflow.contrib.framework.python.ops import audio_ops
#     wav_loader = tf.read_file(example)
#     wav_tensor = audio_ops.decode_wav(wav_loader)
#
#     return wav_tensor


def make_iterator_from_text_dataset(text_dataset, batch_size, unit_dict, shuffle=False, bucket_width=-1, num_cores=4):

    from tensorflow.contrib.lookup import index_table_from_tensor
    table = index_table_from_tensor(mapping=list(unit_dict.values()))

    dataset = tf.data.TextLineDataset(text_dataset)
    dataset = dataset.map(lambda str: tf.string_split([str], delimiter='').values)
    dataset = dataset.map(lambda chars: (chars, tf.size(chars)))
    dataset = dataset.map(lambda chars, size: (table.lookup(chars), size))
    if shuffle is True:
        dataset = dataset.shuffle(buffer_size=1000000, reshuffle_each_iteration=True)

    def batching_fun(x):

        labels_shape = (tf.TensorShape([None]), tf.TensorShape([]), )

        return x.padded_batch(
            batch_size=batch_size,
            padded_shapes=(labels_shape)
        )

    if bucket_width == -1:
        dataset = batching_fun(dataset)
    else:

        def key_func(labels, labels_len):
            # labels_len = tf.shape(labels)[0]
            bucket_id = labels_len // bucket_width
            return tf.cast(bucket_id, dtype=tf.int64)

        def reduce_func(unused_key, windowed_dataset):
            return batching_fun(windowed_dataset)

        dataset = tf.data.Dataset.apply(dataset, tf.data.experimental.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

    dataset = dataset.prefetch(128)

    iterator = dataset.make_initializable_iterator()

    labels, labels_len = iterator.get_next()

    return BatchedData(
        iterator_initializer=iterator.initializer,
        inputs_filenames=None,
        labels_filenames=None,
        inputs=None,
        payload=None,
        inputs_length=None,
        labels=labels,
        labels_length=labels_len
    )

import tensorflow as tf
import collections
from pyVSR.pyVSR.utils import read_htk_file, read_hdf5_file, read_wav_file, file_to_feature
from pyVSR.pyVSR.vsrmath import accurate_derivative
from pyVSR.pyVSR.tcdtimit.files import read_sentence_labels, viseme_list, phoneme_list, character_list
from .audio import process_audio
import numpy as np
from os import path


class BatchedData(collections.namedtuple("BatchedData",
                                         ("initializer",
                                          "filename",
                                          "filename2",
                                          "inputs",
                                          "inputs_len",
                                          "labels",
                                          "labels_len"))):
    pass


def _parse_input_function(example, input_shape, content_type):

    if content_type == 'feature':

        context_features = {
            "input_length": tf.FixedLenFeature(shape=[], dtype=tf.int64),
            "input_size": tf.FixedLenFeature(shape=[], dtype=tf.int64),
            "filename": tf.FixedLenFeature(shape=[], dtype=tf.string)
        }
    elif content_type == 'video':
        context_features = {
            "input_length": tf.FixedLenFeature(shape=[], dtype=tf.int64),
            "width": tf.FixedLenFeature(shape=[], dtype=tf.int64),
            "height": tf.FixedLenFeature(shape=[], dtype=tf.int64),
            "channels": tf.FixedLenFeature(shape=[], dtype=tf.int64),
            "filename": tf.FixedLenFeature(shape=[], dtype=tf.string)
        }
    else:
        raise Exception('unknown content type')

    sequence_features = {
        "inputs": tf.FixedLenSequenceFeature(shape=input_shape, dtype=tf.float32)
    }

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    return sequence_parsed["inputs"], context_parsed["input_length"], context_parsed["filename"]


def _parse_labels_function(example, unit):
    context_features = {
        "unit": tf.FixedLenFeature(shape=[], dtype=tf.string),
        "labels_length": tf.FixedLenFeature(shape=[], dtype=tf.int64),
        "filename": tf.FixedLenFeature(shape=[], dtype=tf.string)
    }
    sequence_features = {
        "labels": tf.FixedLenSequenceFeature(shape=[], dtype=tf.int64)
    }

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    udict = create_unit_dict(unit)
    ivdict = {v: k for k, v in udict.items()}
    labels = tf.concat([sequence_parsed["labels"], [ivdict['EOS']]], axis=0)

    labels_length = context_parsed["labels_length"] + 1

    return labels, labels_length, context_parsed["filename"]


def make_iterator_from_one_record(data_record, label_record, batch_size, shuffle=False, reverse_input=False, bucket_width=-1):

    input_shape, content_type = _get_input_shape_from_record(data_record)
    unit = _get_unit_from_record(label_record)

    dataset1 = tf.data.TFRecordDataset(data_record)
    dataset1 = dataset1.map(lambda proto: _parse_input_function(proto, input_shape, content_type))

    dataset2 = tf.data.TFRecordDataset(label_record)
    dataset2 = dataset2.map(lambda proto: _parse_labels_function(proto, unit))

    dataset = tf.data.Dataset.zip((dataset1, dataset2))

    if shuffle is True:
        dataset = dataset.shuffle(buffer_size=1000)

    if reverse_input is True:
        dataset = dataset.map(
            lambda d1a, d1b, d1c, d2a, d2b, d2c: (tf.reverse(d1a, axis=[0]), d1b, d1c, d2a, d2b, d2c),
        )

    def batching_fun(x):
        return x.padded_batch(
            batch_size=batch_size,
            padded_shapes=(
                (tf.TensorShape([None] + input_shape), tf.TensorShape([]), tf.TensorShape([])),  # input_shape is list
                (tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([]))                 # hence concatenated
            )
        )

    if bucket_width == -1:
        dataset = batching_fun(dataset)
    else:

        def key_func(arg1, arg2):
            inputs_len = tf.shape(arg1[0])[0]
            bucket_id = inputs_len // bucket_width
            return tf.to_int64(bucket_id)

        def reduce_func(unused_key, windowed_dataset):
            return batching_fun(windowed_dataset)

        dataset = tf.data.Dataset.apply(dataset, tf.contrib.data.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

    iterator = dataset.make_initializable_iterator()
    (inputs, inputs_len, fname1), (labels, labels_len, fname2) = iterator.get_next()

    return BatchedData(
        initializer=iterator.initializer,
        filename=fname1,
        filename2=fname2,
        inputs=inputs,
        inputs_len=inputs_len,
        labels=labels,
        labels_len=labels_len
    )


def make_iterator_from_two_records(video_record, audio_record, label_record, batch_size, shuffle=False, reverse_input=False, bucket_width=-1):
    # TODO: this function needs a generalisation to lists of data records

    unit = _get_unit_from_record(label_record)

    vid_input_shape, content_type = _get_input_shape_from_record(video_record)
    vid_dataset = tf.data.TFRecordDataset(video_record)
    vid_dataset = vid_dataset.map(lambda proto: _parse_input_function(proto, vid_input_shape, content_type))

    aud_input_shape, content_type = _get_input_shape_from_record(audio_record)
    aud_dataset = tf.data.TFRecordDataset(audio_record)
    aud_dataset = aud_dataset.map(lambda proto: _parse_input_function(proto, aud_input_shape, content_type))

    dataset1 = tf.data.Dataset.zip((vid_dataset, aud_dataset))

    dataset2 = tf.data.TFRecordDataset(label_record)
    dataset2 = dataset2.map(lambda proto: _parse_labels_function(proto, unit))

    dataset = tf.data.Dataset.zip((dataset1, dataset2))

    if shuffle is True:
        dataset = dataset.shuffle(buffer_size=1000)

    # if reverse_input is True:
    #     dataset = dataset.map(
    #         lambda d1a, d1b, d1c, d2a, d2b, d2c: (tf.reverse(d1a, axis=[0]), d1b, d1c, d2a, d2b, d2c),
    #     )

    def batching_fun(x):
        return x.padded_batch(
            batch_size=batch_size,
            padded_shapes=(
                ((tf.TensorShape([None] + vid_input_shape), tf.TensorShape([]), tf.TensorShape([]), ), (tf.TensorShape([None] + aud_input_shape), tf.TensorShape([]), tf.TensorShape([]), ) ),
                (tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([]))  # hence concatenated
            )
        )

    if bucket_width == -1:
        dataset = batching_fun(dataset)
    else:

        def key_func(arg1, arg2):
            inputs_len = tf.shape(arg1[0][0])[0]
            bucket_id = inputs_len // bucket_width
            return tf.to_int64(bucket_id)

        def reduce_func(unused_key, windowed_dataset):
            return batching_fun(windowed_dataset)

        dataset = tf.data.Dataset.apply(dataset, tf.contrib.data.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

    iterator = dataset.make_initializable_iterator()
    ((vid_inputs, vid_inputs_len, vid_fname1), (aud_inputs, aud_inputs_len, aud_fname1)), (labels, labels_len, fname2) = iterator.get_next()

    return BatchedData(
        initializer=iterator.initializer,
        filename=(vid_fname1, aud_fname1),
        filename2=fname2,
        inputs=(vid_inputs, aud_inputs),
        inputs_len=(vid_inputs_len, aud_inputs_len),
        labels=labels,
        labels_len=labels_len
    )


def _get_input_shape_from_record(record):
    record_iterator = tf.python_io.tf_record_iterator(path=record)

    example = tf.train.SequenceExample()
    example.ParseFromString(record_iterator.__next__())

    # TODO(gsterpu) infer field names from record, avoid try/except

    try:  # one-dimensional input
        input_shape = [example.context.feature["input_size"].int64_list.value[0]]
        content_type = 'feature'

    except:  # two dimensional input
        width = example.context.feature["width"].int64_list.value[0]
        height = example.context.feature["height"].int64_list.value[0]

        try:
            channels = example.context.feature["channels"].int64_list.value[0]
        except:
            channels = 1

        input_shape = [width, height, channels]
        content_type = 'video'

    return input_shape, content_type


def _get_unit_from_record(record):
    record_iterator = tf.python_io.tf_record_iterator(path=record)

    example = tf.train.SequenceExample()
    example.ParseFromString(record_iterator.__next__())
    unit = example.context.feature["unit"].bytes_list.value[0]
    return unit.decode('utf-8')


def _symbols_to_ints(symbols, unit):

    unit_dict = create_unit_dict(unit)
    ivdict = {v: k for k, v in unit_dict.items()}

    ints = [ivdict[symbol] for symbol in symbols]
    # ints.append(ivdict['EOS'])  # EOS is now appended dynamically by the iterator
    return np.asarray(ints, dtype=np.int32)


def create_unit_dict(unit):

    unit_dict = {'MASK': 0, 'END': -1}

    unit_list = _read_unit_file(unit)
    idx = 0
    for idx, subunit in enumerate(unit_list):
        unit_dict[subunit] = idx + 1

    unit_dict['EOS'] = idx + 2
    unit_dict['GO'] = idx + 3

    ivdict = {v: k for k, v in unit_dict.items()}

    return ivdict


def _read_unit_file(unit):
    if unit == 'viseme':
        myfile = viseme_list
    elif unit == 'phoneme':
        myfile = phoneme_list
    elif unit == 'character':
        myfile = character_list
    else:
        raise Exception('unallowed unit')

    with open(myfile, 'r') as f:
        contents = f.read().splitlines()

    return contents


def write_dataset(files, content_type, outpath, file_type=None, transformation=None):
    r"""

    Parameters
    ----------
    files
    file_type: `htk` or `hdf5`
    content_type: `video`(3D/4D array) or `feature` (1D array) or 'labels` (0D array)
    outpath

    Returns
    -------

    """

    if content_type in ('viseme', 'phoneme', 'character'):
        write_label_dataset(files, content_type, outpath)

    elif content_type in ('feature', 'video'):
        write_input_dataset(files, content_type, file_type, outpath, transformation)


def write_input_dataset(files, content_type, file_type, outpath, transformation):
    writer = tf.python_io.TFRecordWriter(outpath)

    for file in files:
        print(file)

        contents = read_data_file(file, file_type)

        if transformation is not None:
            contents = apply_transform(contents, transformation)

        example = make_input_example(file, contents, content_type)

        writer.write(example.SerializeToString())

    writer.close()


def apply_transform(data, transformation):
    if transformation == 'mfcc':
        return wav_to_mfcc(data)

    elif transformation == 'mfcc_d_a':
        mfcc = wav_to_mfcc(data)
        delta = accurate_derivative(mfcc, derivative_type='delta')
        double_delta = accurate_derivative(mfcc, derivative_type='double_delta')

        return np.hstack((mfcc, delta, double_delta))

    else:
        raise Exception('unsupported transformation')


def wav_to_mfcc(wav):

    wav_tensor = tf.convert_to_tensor(wav,
                                      dtype=tf.float32)

    hparams_audio = tf.contrib.training.HParams(
        frame_length_msec=25,  # 25 > 20
        frame_step_msec=10,
        sample_rate=22050,
        fft_length=1024,
        mel_lower_edge_hz=80,
        mel_upper_edge_hz=22050 / 2,  # 11025 > 7600
        num_mel_bins=30,  # 30 > 60 > 80
        num_mfccs=26,  # 26 > 13
    )

    mfcc_tensor = process_audio(wav_tensor, hparams_audio)

    session_conf = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 0},
        allow_soft_placement=False,
        log_device_placement=False
    )

    sess = tf.InteractiveSession(config=session_conf)
    # print(len(sess.graph.get_operations()))  # debug graph ops
    mfcc = mfcc_tensor.eval()

    sess.close()
    tf.reset_default_graph()

    return mfcc


def write_label_dataset(files, unit, outpath):
    writer = tf.python_io.TFRecordWriter(outpath)
    for file in files:
        ground_truth = np.asarray(read_sentence_labels(file, unit))
        labels = _symbols_to_ints(ground_truth, unit)

        example = make_label_example(file, labels, unit)
        writer.write(example.SerializeToString())

    writer.close()


def read_data_file(file, filetype):
    if filetype == 'htk':
        contents = read_htk_file(file)
    elif filetype == 'hdf5':
        contents = read_hdf5_file(file)
    elif filetype == 'wav':
        contents = read_wav_file(file)
    else:
        raise Exception('unknown file type')

    return contents


def make_input_example(file, data, content_type):
    if content_type == 'feature':
        example = make_feature_example(file, data)
    elif content_type == 'video':
        example = make_video_example(file, data)
    else:
        raise Exception('unknown content type')

    return example


def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_feature_list(values_list):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v.tostring()) for v in values_list])


def make_feature_example(file, inputs):

    input_steps = len(inputs)
    input_size = inputs.shape[-1]

    base, ext = path.splitext(file)
    if ext == '.wav':
        filename = file_to_feature(file, extension='')
    else:
        filename = path.splitext(path.split(file)[-1])[0]

    context = tf.train.Features(feature={
        "input_length": _int64_feature(input_steps),
        "input_size": _int64_feature(input_size),
        "filename": _bytes_feature(bytes(filename, encoding='utf-8'))})

    input_features = [
        tf.train.Feature(float_list=tf.train.FloatList(value=input_))
        for input_ in inputs]

    feature_list = {
        'inputs': tf.train.FeatureList(feature=input_features)
    }
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    return tf.train.SequenceExample(feature_lists=feature_lists,
                                    context=context)


def make_video_example(file, frames):

    filename = path.splitext(path.split(file)[-1])[0]
    input_len = len(frames)

    ndim = np.ndim(frames[0])
    if ndim == 2:
        width, height = np.shape(frames[0])
        channels = 1
    elif ndim == 3:
        width, height, channels = np.shape(frames[0])
    else:
        raise Exception('unsupported number of video channels')

    context = tf.train.Features(feature={
        "input_length": _int64_feature(input_len),
        "width": _int64_feature(width),
        "height": _int64_feature(height),
        "channels": _int64_feature(channels),
        "filename": _bytes_feature(bytes(filename, encoding='utf-8'))})

    input_features = [
        tf.train.Feature(float_list=tf.train.FloatList(value=frame.flatten()))
        for frame in frames]

    feature_list = {
        'inputs': tf.train.FeatureList(feature=input_features)
    }
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    return tf.train.SequenceExample(feature_lists=feature_lists,
                                    context=context)


def make_label_example(file, labels, unit):

    filename = path.splitext(path.split(file)[-1])[0]
    labels_len = len(labels)

    context = tf.train.Features(feature={
        "unit": _bytes_feature(bytes(unit, encoding='utf-8')),
        "labels_length": _int64_feature(labels_len),
        "filename": _bytes_feature(bytes(filename, encoding='utf-8'))
    })

    label_features = [
        tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        for label in labels]

    feature_list = {
        'labels': tf.train.FeatureList(feature=label_features)
    }

    feature_lists = tf.train.FeatureLists(feature_list=feature_list)

    return tf.train.SequenceExample(feature_lists=feature_lists,
                                    context=context)


# Possible feature: decode files on the fly
#
# def parse_files_function(example):
#     from tensorflow.contrib.framework.python.ops import audio_ops
#     wav_loader = tf.read_file(example)
#     wav_tensor = audio_ops.decode_wav(wav_loader)
#
#     return wav_tensor
#
#
# def make_iterator_from_filenames(files, batch_size, shuffle=False, reverse_input=False, bucket_width=-1, unit='phoneme'):
#     labels = []
#     for file in files:
#         ground_truth = np.asarray(read_sentence_labels(file, unit))
#         labels.append(_symbols_to_ints(ground_truth, unit))
#
#     order = tf.range(len(labels))
#     d2 = tf.data.Dataset.from_tensor_slices(order)
#     d2 = d2.map(lambda i: (files[i], labels[i]))
#
#
#     dataset = tf.data.Dataset.from_tensor_slices(files)
#     dataset = dataset.map(parse_files_function, num_parallel_calls=batch_size)

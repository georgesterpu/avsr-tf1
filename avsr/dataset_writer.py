import tensorflow as tf
import numpy as np
from .audio import process_audio, read_wav_file
from os import path, makedirs
import glob
from imageio import imread
import cv2
from .awgn import cache_noise, add_noise_cached


class TFRecordWriter(object):

    def __init__(self,
                 train_files,
                 test_files,
                 label_map):

        self._train_files = _remove_extensions(train_files)
        self._test_files = _remove_extensions(test_files)
        self._label_map = label_map

    def write_labels_records(self,
                             unit,
                             unit_list_file,
                             label_file,
                             train_record_name,
                             test_record_name):

        labels_dict = _create_labels_dict(label_file)
        unit_dict = _create_unit_dict(unit_list_file)

        files = (self._train_files, self._test_files)
        for idx, record in enumerate([train_record_name, test_record_name]):
            makedirs(path.dirname(record), exist_ok=True)
            writer = tf.python_io.TFRecordWriter(record)

            for file in files[idx]:
                print(file)
                sentence_id = self._label_map[file]
                ground_truth = labels_dict[sentence_id]
                labels = _symbols_to_ints(ground_truth, unit_dict)

                example = _make_label_example(sentence_id, labels, unit)
                writer.write(example.SerializeToString())

            writer.close()

    def write_audio_records(self,
                            train_record_name,
                            test_record_name,
                            content_type=None,
                            extension=None,
                            transform=None,
                            noise_type=None,
                            snr_list=(),
                            target_sr=22050,
                            ):

        files = (self._train_files,
                 self._test_files,)

        if transform is not None:
            # build the graph with tf ops only once
            # still sub-optimal since batch size will be one
            engine = _build_audio_engine(target_sr=target_sr, transformation=transform)

        # preload noise data
        if len(snr_list) > 0:
            noise_data = cache_noise(noise_type, sampling_rate=target_sr)
        else:
            noise_data = None
            snr_list = ('clean', )

        for idx, record in enumerate([train_record_name, test_record_name]):

            writers = []
            for snr in snr_list:
                if snr == 'clean':
                    record_name = path.join(record + '_' + str(snr) + '.tfrecord', )
                else:
                    record_name = path.join(record + '_' + noise_type + '_' + str(snr) + 'db.tfrecord', )
                writers.append(tf.python_io.TFRecordWriter(record_name))

            for file in files[idx]:
                print(file)
                sentence_id = self._label_map[file]

                input_data = read_data_file(file, extension, sr=target_sr)

                for snr_idx, snr in enumerate(snr_list):

                    data = np.copy(input_data)  # safety first ? we don't have const in Python

                    if snr is not 'clean':
                        data = add_noise_cached(  # this is the function we don't trust
                            orig_signal=data,
                            noise_type=noise_type,
                            noise_data=noise_data,
                            snr=snr,)

                    if transform is not None:
                        transformed_data = apply_transform(data=data, transformation=transform, engine=engine)
                    else:
                        transformed_data = data

                    example = make_input_example(sentence_id, transformed_data, content_type)

                    writers[snr_idx].write(example.SerializeToString())

            for writer in writers:
                writer.close()

        if transform is not None:
            engine.sess.close()

    def write_video_records(self,
                            train_record_name,
                            test_record_name,
                            content_type=None,
                            extension=None,
                            ):
        files = (self._train_files,
                 self._test_files,)

        for idx, record in enumerate([train_record_name, test_record_name]):
            makedirs(path.dirname(record), exist_ok=True)
            writer = tf.python_io.TFRecordWriter(record)

            for file in files[idx]:
                # TODO add progress info
                print(file)
                sentence_id = self._label_map[file]

                contents = read_data_file(file, extension)

                example = make_input_example(sentence_id, contents, content_type)

                writer.write(example.SerializeToString())

            writer.close()

    def write_bmp_records(self,
                          train_record_name,
                          test_record_name,
                          bmp_dir,
                          output_resolution,
                          crop_lips=False):

        files = (self._train_files,
                 self._test_files,)

        for idx, record in enumerate([train_record_name, test_record_name]):
            writer = tf.python_io.TFRecordWriter(record)

            for file in files[idx]:
                # TODO add progress info
                print(file)
                sentence_id = self._label_map[file]
                feature_dir = path.join(bmp_dir, sentence_id) + '_aligned'

                contents = read_bmp_dir(feature_dir, output_resolution, crop_lips)

                example = make_input_example(sentence_id, contents, 'video')

                writer.write(example.SerializeToString())

            writer.close()


def _create_labels_dict(file):
    with open(file, 'r') as f:
        contents = f.read().splitlines()

    labels_dict = dict([line.split(' ', maxsplit=1) for line in contents])

    return labels_dict


def _create_unit_dict(unit_list_file):

    unit_dict = {'MASK': 0, 'END': -1}

    with open(unit_list_file, 'r') as f:
        unit_list = f.read().splitlines()

    idx = 0
    for idx, subunit in enumerate(unit_list):
        unit_dict[subunit] = idx + 1

    unit_dict['EOS'] = idx + 2
    unit_dict['GO'] = idx + 3

    # ivdict = {v: k for k, v in unit_dict.items()}

    return unit_dict


def _symbols_to_ints(symbols, unit_dict):

    ints = [unit_dict[symbol] for symbol in symbols]
    return np.asarray(ints, dtype=np.int32)


def _make_label_example(label_id, labels, unit):

    labels_len = len(labels)

    context = tf.train.Features(feature={
        "unit": _bytes_feature(bytes(unit, encoding='utf-8')),
        "labels_length": _int64_feature(labels_len),
        "filename": _bytes_feature(bytes(label_id, encoding='utf-8'))
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


def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_feature_list(values_list):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v.tostring()) for v in values_list])


def read_data_file(file, extension, sr=None):
    if extension in ('wav', 'mp4', 'WAV'):
        contents = read_wav_file(file + '.' + extension, sr=sr)
    else:
        raise Exception('unknown file type/extension')

    # elif extension == 'htk':
    #     from pyVSR.pyVSR.utils import read_htk_file
    #     contents = read_htk_file(file)
    # elif extension == 'h5':
    #     from pyVSR.pyVSR.utils import read_hdf5_file
    #     contents = read_hdf5_file(file + '.' + extension)

    return contents


def apply_transform(data, transformation, engine):
    data = np.squeeze(data, axis=-1)
    if transformation == 'mfcc':
        return eval_audio_model(wav=data, engine=engine)

    elif transformation == 'mfcc_d_a':
        mfcc = eval_audio_model(wav=data, engine=engine)
        delta = np.diff(mfcc, n=1, axis=0)
        double_delta = np.diff(mfcc, n=2, axis=0)

        return np.hstack((mfcc, delta, double_delta))

    elif transformation == 'logmel':
        return eval_audio_model(wav=data, engine=engine)

    elif transformation == 'logmel_d_a':
        logmel = eval_audio_model(wav=data, engine=engine)

        delta = np.diff(logmel, n=1, axis=0)
        delta = np.vstack((delta, delta[-1, :]))

        double_delta = np.diff(logmel, n=2, axis=0)
        double_delta = np.vstack((double_delta, double_delta[-1, :], double_delta[-1, :]))

        return np.hstack((logmel, delta, double_delta))

    elif transformation == 'logmel_stack_w8s3':
        logmel = eval_audio_model(wav=data, engine=engine)
        logmel = _stack_features(logmel, window_len=8, stride=3)

        return logmel

    else:
        raise Exception('unsupported transformation')


def eval_audio_model(wav, engine):

    feature = engine.sess.run(engine.output_tensor, feed_dict={engine.input_tensor: wav})
    return feature


def _build_audio_engine(target_sr, transformation):
    from collections import namedtuple
    engine = namedtuple('Engine', ['sess', 'input_tensor', 'output_tensor'])

    graph = tf.Graph()

    with graph.as_default():
        input_tensor = tf.placeholder(dtype=tf.float32, shape=(None,))

        # wav_tensor = tf.convert_to_tensor(input,
        #                                   dtype=tf.float32)

        hparams_audio = tf.contrib.training.HParams(
            frame_length_msec=25,  # 25 > 20
            frame_step_msec=10,
            sample_rate=target_sr,
            fft_length=1024,
            mel_lower_edge_hz=80,
            mel_upper_edge_hz=target_sr / 2,  # 11025 > 7600
            num_mel_bins=30,  # 30 > 60 > 80
            num_mfccs=26,  # 26 > 13
        )

        features = process_audio(input_tensor, hparams_audio, logmel_only=('logmel' in transformation))

    session_conf = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 0},
        allow_soft_placement=False,
        log_device_placement=False
    )

    sess = tf.Session(config=session_conf, graph=graph)
    # print(len(sess.graph.get_operations()))  # debug graph ops

    return engine(sess=sess, input_tensor=input_tensor, output_tensor=features)


def make_input_example(file, data, content_type):
    if content_type == 'feature':
        example = make_feature_example(file, data)
    elif content_type == 'video':
        example = make_video_example(file, data)
    else:
        raise Exception('unknown content type')

    return example


def make_feature_example(sentence_id, inputs):

    input_steps = len(inputs)
    input_size = inputs.shape[-1]

    context = tf.train.Features(feature={
        "input_length": _int64_feature(input_steps),
        "input_size": _int64_feature(input_size),
        "filename": _bytes_feature(bytes(sentence_id, encoding='utf-8'))})

    input_features = [
        tf.train.Feature(float_list=tf.train.FloatList(value=input_))
        for input_ in inputs]

    feature_list = {
        'inputs': tf.train.FeatureList(feature=input_features)
    }
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    return tf.train.SequenceExample(feature_lists=feature_lists,
                                    context=context)


def make_video_example(sentence_id, frames):

    input_len = len(frames)

    ndim = np.ndim(frames[0])
    if ndim == 2:
        height, width = np.shape(frames[0])
        channels = 1
    elif ndim == 3:
        height, width, channels = np.shape(frames[0])
    else:
        raise Exception('unsupported number of video channels')

    context = tf.train.Features(feature={
        "input_length": _int64_feature(input_len),
        "width": _int64_feature(width),
        "height": _int64_feature(height),
        "channels": _int64_feature(channels),
        "filename": _bytes_feature(bytes(sentence_id, encoding='utf-8'))})

    input_features = [
        tf.train.Feature(float_list=tf.train.FloatList(value=frame.flatten()))
        for frame in frames]

    feature_list = {
        'inputs': tf.train.FeatureList(feature=input_features)
    }
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    return tf.train.SequenceExample(feature_lists=feature_lists,
                                    context=context)


def _remove_extensions(filenames):
    return [path.splitext(file)[0] for file in filenames]


def _add_extensions(filenames, extension):
    return [path.join(file + '.' + extension) for file in filenames]


def read_bmp_dir(feature_dir, output_resolution, crop_lips=False):
    files = sorted(glob.glob(feature_dir + '/*.bmp'))
    data = []
    for file in files:
        image = imread(file)
        rows, cols, nchan = image.shape
        if crop_lips is True:
            image = image[(3*rows//5):, (1*cols//10):(9*cols//10), :]
        resized = cv2.resize(image, output_resolution, interpolation=cv2.INTER_AREA,)  # area better when decimating
        data.append(resized)

    video = np.asarray(data, dtype=np.float64)

    # for frame in video:
    #     frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_AREA)
    #     cv2.imshow('video_stream', cv2.cvtColor(frame, code=cv2.COLOR_RGB2BGR))
    #     cv2.waitKey(30)
    video = (video - 128) / 128

    return video


def _stack_features(mat, window_len, stride):
    nrows = ((mat.shape[0]-window_len)//stride)+1
    return mat[stride*np.arange(nrows)[:, None] + np.arange(window_len)].reshape([nrows, -1])

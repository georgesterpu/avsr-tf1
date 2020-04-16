from os import path
import os
from avsr.dataset_writer import TFRecordWriter
from avsr.utils import get_files

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"  # ERROR


def main():

    dataset_dir = '/run/media/john_tukey/download/datasets/tcdtimit/'
    train_list = './datasets/tcdtimit/splits/speaker-dependent/train.scp'
    test_list = './datasets/tcdtimit/splits/speaker-dependent/test.scp'

    train = get_files(train_list, dataset_dir)
    test = get_files(test_list, dataset_dir)

    label_map = dict()
    for file in train+test:
        label_map[path.splitext(file)[0]] = path.splitext(file.split('tcdtimit/')[-1])[0]

    writer = TFRecordWriter(
        train_files=train,
        test_files=test,
        label_map=label_map,
        )

    writer.write_labels_records(
        unit='character',
        unit_list_file='./avsr/misc/character_list',
        label_file='./datasets/tcdtimit/configs/character_labels',
        train_record_name='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords/characters_train_sd.tfrecord',
        test_record_name='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords/characters_test_sd.tfrecord',
    )

    writer.write_audio_records(
        content_type='feature',
        extension='wav',
        transform='logmel_stack_w8s3',
        snr_list=['clean', 10, 0, -5],
        target_sr=16000,
        noise_type='cafe',
        train_record_name='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords/logmel_train_sd',
        test_record_name='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords/logmel_test_sd',
    )

    writer.write_bmp_records(
        train_record_name='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords/rgb36lips_train_sd.tfrecord',
        test_record_name='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords/rgb36lips_test_sd.tfrecord',
        bmp_dir='/run/media/john_tukey/download/datasets/tcdtimit/aligned_openface/',
        output_resolution=(36, 36),
        crop_lips=True,
    )


if __name__ == '__main__':
    main()


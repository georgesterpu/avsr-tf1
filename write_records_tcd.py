from os import path
import os
from avsr.dataset_writer import TFRecordWriter
# from datasets.tcdtimit.files import request_files
from .extract_faces import get_files

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"  # ERROR


def main():

    dataset_dir = '/run/media/john_tukey/download/datasets/adapt/tcdtimit/'
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
        unit_list_file='/run/media/john_tukey/download/datasets/MV-LRS/misc/character_list',
        label_file='/run/media/john_tukey/work/phd/32.seq2seq_feature/myseq2seq/pyVSR/pyVSR/tcdtimit/htkconfigs/character_labels',
        train_record_name='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords4/characters_train_sd.tfrecord',
        test_record_name='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords4/characters_test_sd.tfrecord',
    )

    writer.write_audio_records(
        content_type='feature',
        extension='wav',
        transform='logmel_stack_w8s3',
        # snr_list=[10, 0, -5],
        target_sr=22050,
        # noise_type='street',
        train_record_name='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords4/logmel_train_sd_stack',
        test_record_name='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords4/logmel_test_sd_stack',
    )

    writer.write_bmp_records(
        train_record_name='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords4/rgb36lips_train_sd.tfrecord',
        test_record_name='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords4/rgb36lips_test_sd.tfrecord',
        bmp_dir='/run/media/john_tukey/download/datasets/tcdtimit/aligned_openface/',
        output_resolution=(36, 36),
        crop_lips=True,
    )


if __name__ == '__main__':
    main()


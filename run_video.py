import os
from avsr import run_experiment

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"  # ERROR


def main():
    records_path = './data/'
    video_train_record = records_path + 'rgb36lips_train_success_aus.tfrecord'
    video_test_record = records_path + 'rgb36lips_test_success_aus.tfrecord'
    labels_train_record = records_path + 'characters_train_success.tfrecord'
    labels_test_record = records_path + 'characters_test_success.tfrecord'

    iterations = (
        (100, 20),  # clean
    )

    learning_rates = (
        (0.001, 0.0001),  # clean
    )

    logfile = 'lrs2_vid2chars'

    run_experiment(
        video_train_record=video_train_record,
        video_test_record=video_test_record,
        labels_train_record=labels_train_record,
        labels_test_record=labels_test_record,
        iterations=iterations,
        learning_rates=learning_rates,
        architecture='unimodal',
        logfile=logfile,
        video_processing='resnet_cnn',
        input_modality='video',
        regress_aus=True,
    )


if __name__ == '__main__':
    main()

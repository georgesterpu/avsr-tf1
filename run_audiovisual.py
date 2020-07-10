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

    audio_train_records = (
        records_path + 'logmel_train_success_clean.tfrecord',
        records_path + 'logmel_train_success_cafe_10db.tfrecord',
        records_path + 'logmel_train_success_cafe_0db.tfrecord',
        records_path + 'logmel_train_success_cafe_-5db.tfrecord'
    )

    audio_test_records = (
       records_path + 'logmel_test_success_clean.tfrecord',
       records_path + 'logmel_test_success_cafe_10db.tfrecord',
       records_path + 'logmel_test_success_cafe_0db.tfrecord',
       records_path + 'logmel_test_success_cafe_-5db.tfrecord'
    )

    iterations = (
        (100, 20),  # clean
        (100, 20),  # 10db
        (100, 20),  # 0db
        (100, 20)     # -5db
    )

    learning_rates = (
        (0.001, 0.0001),  # clean
        (0.001, 0.0001),  # 10db
        (0.001, 0.0001),  # 0db
        (0.001, 0.0001)       # -5db
    )

    logfile = 'lrs2_avalign'

    run_experiment(
        video_train_record=video_train_record,
        video_test_record=video_test_record,
        labels_train_record=labels_train_record,
        labels_test_record=labels_test_record,
        audio_train_records=audio_train_records,
        audio_test_records=audio_test_records,
        iterations=iterations,
        learning_rates=learning_rates,
        architecture='av_align',
        regress_aus=True,
        audio_processing='features',
        video_processing='resnet_cnn',
        logfile=logfile,
    )


if __name__ == '__main__':
    main()

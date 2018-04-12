import avsr
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def main():

    experiment = avsr.AVSR(
        unit='viseme',
        video_processing='resnet_cnn',
        cnn_filters=(8, 16, 24, 32),
        cnn_dense_units=128,
        video_train_record='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords2/rgb36lips_train_sd.tfrecord',
        video_test_record='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords2/rgb36lips_test_sd.tfrecord',
        audio_processing='features',
        audio_train_record='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords2/logmel_train_sd.tfrecord',
        audio_test_record='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords2/logmel_test_sd.tfrecord',
        labels_train_record = '/run/media/john_tukey/download/datasets/tcdtimit/tfrecords2/visemes_train_sd.tfrecord',
        labels_test_record = '/run/media/john_tukey/download/datasets/tcdtimit/tfrecords2/visemes_test_sd.tfrecord',
        encoder_type='unidirectional',
        decoding_algorithm='beam_search',
        encoder_units_per_layer=(128, 128, ),
        decoder_units_per_layer=(256, ),
        batch_size=(32, 64),
        learning_rate=0.001,
        label_skipping=False,
    )

    experiment.train(
        num_epochs=401,
        logfile='./logs/new',
        try_restore_latest_checkpoint=False
    )



if __name__ == '__main__':
    main()

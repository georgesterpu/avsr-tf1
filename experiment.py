import avsr
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def main():

    experiment = avsr.AVSR(
        unit='viseme',
        video_processing='resnet_cnn',
        cnn_filters=(16, 32, 64),
        cnn_dense_units=128,
        video_train_record='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords/aligned/roi_rgb_36_train.tfrecord',
        video_test_record='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords/aligned/roi_rgb_36_test.tfrecord',
        # video_train_record='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords_icassp2018/dct_train_v3.tfrecord',
        # video_test_record='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords_icassp2018/dct_test_v3.tfrecord',
        audio_processing='features',
        audio_train_record='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords_2018/mfcc_train.tfrecord',
        audio_test_record='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords_2018/mfcc_test.tfrecord',
        labels_train_record = '/run/media/john_tukey/download/datasets/tcdtimit/tfrecords/visemes_train.tfrecord',
        labels_test_record = '/run/media/john_tukey/download/datasets/tcdtimit/tfrecords/visemes_test.tfrecord',
        # labels_train_record='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords/characters_train.tfrecord',
        # labels_test_record='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords/characters_test.tfrecord',
        encoder_type='unidirectional',
        decoding_algorithm='beam_search',
        encoder_units_per_layer=(128, 128, ),
        decoder_units_per_layer=(256, ),
        batch_size=(32, 32),
        learning_rate=0.001,
        label_skipping=False,
    )

    experiment.train(
        num_epochs=401,
        logfile='./logs/test_iter_data',
        try_restore_latest_checkpoint=False
    )

    # corr, acc = experiment.evaluate(
    #     checkpoint_path='./checkpoints/a_only/checkpoint.ckp-10',
    #     epoch=69,
    # )
    # print('Accuracy: {}'.format(acc))


if __name__ == '__main__':
    main()

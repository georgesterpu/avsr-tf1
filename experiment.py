import avsr
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"  # ERROR


def main():

    experiment = avsr.AVSR(
        unit='viseme',
        unit_file='/run/media/john_tukey/download/datasets/MV-LRS/misc/viseme_list',
        video_processing='resnet_cnn',
        cnn_filters=(8, 16, 24, 32),
        cnn_dense_units=128,
        video_train_record='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords2/rgb36lips_train_sd.tfrecord',
        video_test_record='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords2/rgb36lips_test_sd.tfrecord',
        audio_processing='features',
        audio_train_record='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords2/logmel_train_sd_0db.tfrecord',
        audio_test_record='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords2/logmel_test_sd_0db.tfrecord',
        labels_train_record = '/run/media/john_tukey/download/datasets/tcdtimit/tfrecords2/visemes_train_sd.tfrecord',
        labels_test_record = '/run/media/john_tukey/download/datasets/tcdtimit/tfrecords2/visemes_test_sd.tfrecord',
        encoder_type='unidirectional',
        decoding_algorithm='beam_search',
        encoder_units_per_layer=(128, 128, ),
        decoder_units_per_layer=(256, ),
        attention_type=(('scaled_luong', )*1, ('scaled_luong', )*1),
        batch_size=(32, 64),
        learning_rate=0.0001,
        label_skipping=False,
    )

    # uer = experiment.evaluate(
    #     checkpoint_path='./checkpoints/sd_new2_multiattn/checkpoint.ckp-31',
    # )
    # print(uer)
    # return

    experiment.train(
        num_epochs=201,
        logfile='./logs/tcd_sd_visemes_allnoise_r11',
        try_restore_latest_checkpoint=True
    )



if __name__ == '__main__':
    main()

import avsr
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"  # ERROR


def main():

    experiment = avsr.AVSR(
        unit='character',
        unit_file='/run/media/john_tukey/download/datasets/MV-LRS/misc/character_list',
        video_processing='resnet_cnn',
        cnn_filters=(8, 16, 24, 32),
        cnn_dense_units=128,
        video_train_record='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords2/rgb36lips_train_sd.tfrecord',
        video_test_record='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords2/rgb36lips_test_sd.tfrecord',
        audio_processing='features',
        audio_train_record='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords2/logmel_train_sd_10db.tfrecord',
        audio_test_record='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords2/logmel_test_sd_10db.tfrecord',
        labels_train_record = '/run/media/john_tukey/download/datasets/tcdtimit/tfrecords2/characters_train_sd.tfrecord',
        labels_test_record = '/run/media/john_tukey/download/datasets/tcdtimit/tfrecords2/characters_test_sd.tfrecord',
        encoder_type='unidirectional',
        decoding_algorithm='beam_search',
        encoder_units_per_layer=(256, 256, 256),
        decoder_units_per_layer=(256, ),
        attention_type=(('scaled_luong', )*3, ('scaled_luong', )*1),
        batch_size=(64, 64),
        optimiser='AMSGrad',
        learning_rate=0.001,
        label_skipping=False,
    )

    # uer = experiment.evaluate(
    #     checkpoint_path='./checkpoints/sd_new2_multiattn/checkpoint.ckp-31',
    # )
    # print(uer)
    # return

    experiment.train(
        num_epochs=91,
        logfile='./logs/tcd_sd_characters_wgn_r31_3x256',
        try_restore_latest_checkpoint=True
    )



if __name__ == '__main__':
    main()

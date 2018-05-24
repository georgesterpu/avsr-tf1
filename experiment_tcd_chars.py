import avsr
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"  # ERROR
import sys

def main(argv):

    num_epochs = int(argv[1])
    learning_rate = float(argv[2])

    experiment = avsr.AVSR(
        unit='character',
        unit_file='/run/media/john_tukey/download/datasets/MV-LRS/misc/character_list',
        video_processing='resnet_cnn',
        cnn_filters=(8, 16, 32, 64),
        cnn_dense_units=128,
        video_train_record='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords2/rgb36lips_train_sd.tfrecord',
        video_test_record='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords2/rgb36lips_test_sd.tfrecord',
        audio_processing='features',
        audio_train_record='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords2/logmel_train_sd_stack_clean.tfrecord',
        audio_test_record='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords2/logmel_test_sd_stack_clean.tfrecord',
        labels_train_record ='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords2/characters_train_sd.tfrecord',
        labels_test_record ='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords2/characters_test_sd.tfrecord',
        encoder_type='unidirectional',
        architecture='av_align',
        decoding_algorithm='beam_search',
        encoder_units_per_layer=(256, 256, 256),
        decoder_units_per_layer=(256, ),
        attention_type=(('scaled_luong', )*3, ('scaled_luong', )),
        batch_size=(64, 64),
        optimiser='AMSGrad',
        learning_rate=learning_rate,
        label_skipping=False,
        num_gpus=1,
    )

    # uer = experiment.evaluate(
    #     checkpoint_path='./checkpoints/sd_new2_multiattn/checkpoint.ckp-31',
    # )
    # print(uer)
    # return

    experiment.train(
        num_epochs=num_epochs,
        logfile='./logs/mwer_audio_attempt1',
        try_restore_latest_checkpoint=True
    )



if __name__ == '__main__':
    main(sys.argv)

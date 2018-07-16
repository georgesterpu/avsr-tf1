import avsr
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"  # ERROR
import sys


def main(argv):

    num_epochs = int(argv[1])
    learning_rate = float(argv[2])

    experiment = avsr.AVSR(
        unit='character',
        unit_file='/run/media/john_tukey/download/datasets/MV-LRS/misc/character_list',
        video_processing='resnet_cnn',
        #cnn_filters=(12, 32, 64, 96),
        cnn_filters=(8, 16, 32, 64),
        cnn_dense_units=16,
        batch_normalisation=True,
        video_train_record='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords2/rgb36lips_train_sd.tfrecord',
        video_test_record='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords2/rgb36lips_test_sd.tfrecord',
        audio_processing=None,
        audio_train_record='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords2/logmel_train_sd_stack_clean.tfrecord',
        audio_test_record='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords2/logmel_test_sd_stack_clean.tfrecord',
        labels_train_record ='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords2/characters_train_sd.tfrecord',
        labels_test_record ='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords2/characters_test_sd.tfrecord',
        encoder_type='unidirectional',
        architecture='unimodal',
        clip_gradients=True,
        max_gradient_norm=1.0,
        recurrent_regularisation=0.00001,
        cell_type='gru',
        residual_encoder=True,
        sampling_probability_outputs=0.1,
        #dropout_probability=(0.7, 0.7, 0.7),
        decoding_algorithm='beam_search',
        encoder_units_per_layer=(256, 256, 256,),
        decoder_units_per_layer=(256, ),
        attention_type=(('scaled_luong', )*3, ('scaled_luong', )*3),
        mwer_training=False,
        beam_width=10,
        batch_size=(48, 64),
        optimiser='AMSGrad',
        learning_rate=learning_rate,
        label_skipping=False,
        num_gpus=1,
    )

    #uer = experiment.evaluate(
    #   checkpoint_path='./checkpoints/video_to_chars_gru_3x256_residual_00001reg/checkpoint.ckp-1005',
    #)
    #print(uer)
    #return

    experiment.train(
        num_epochs=num_epochs,
        logfile='./logs/tcd_vid_to_chars_sd_16denseunits',
        try_restore_latest_checkpoint=True
    )



if __name__ == '__main__':
    main(sys.argv)

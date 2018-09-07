import sys
import os
import avsr

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"  # ERROR


def main(argv):

    num_epochs = int(argv[1])
    learning_rate = float(argv[2])

    experiment = avsr.AVSR(
        unit='character',
        unit_file='./avsr/misc/character_list',
        video_processing=None,
        batch_normalisation=True,
        audio_processing='features',
        audio_train_record='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords4/logmel_train_sd_stack_clean.tfrecord',
        audio_test_record='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords4/logmel_test_sd_stack_clean.tfrecord',
        labels_train_record ='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords4/characters_train_sd.tfrecord',
        labels_test_record ='/run/media/john_tukey/download/datasets/tcdtimit/tfrecords4/characters_test_sd.tfrecord',
        encoder_type='unidirectional',
        architecture='unimodal',
        clip_gradients=True,
        max_gradient_norm=1.0,
        recurrent_regularisation=0.0001,
        cell_type='gru',
        embedding_size=128,
        highway_encoder=False,
        sampling_probability_outputs=0.1,
        use_dropout=True,
        dropout_probability=(0.9, 0.9, 0.9),
        decoding_algorithm='beam_search',
        enable_attention=True,
        encoder_units_per_layer=(128, 128, ),
        decoder_units_per_layer=(128, ),
        attention_type=(('scaled_luong', )*1, ('scaled_luong', )*1),
        beam_width=10,
        batch_size=(64, 64),
        optimiser='AMSGrad',
        learning_rate=learning_rate,
        num_gpus=1,
    )

    # uer = experiment.evaluate(
    #   checkpoint_path='./checkpoints/tcd_audio_to_chars/checkpoint.ckp-400',
    # )
    # print(uer)
    # return

    experiment.train(
        num_epochs=num_epochs,
        logfile='./logs/tcd_audio_to_chars',
        try_restore_latest_checkpoint=True
    )


if __name__ == '__main__':
    main(sys.argv)

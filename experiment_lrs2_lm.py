import os
import avsr


os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"  # ERROR


def main():

    experiment = avsr.LM(
        unit='character',
        unit_file='./avsr/misc/character_list',
        labels_train_record ='/run/media/john_tukey/download/datasets/lrs2/tfrecords/characters_train_success.tfrecord',
        labels_test_record ='/run/media/john_tukey/download/datasets/lrs2/tfrecords/characters_test_success.tfrecord',
        clip_gradients=True,
        decoder_units_per_layer=(256, ),
        learning_rate=0.001,
    )

    experiment.train(
        num_epochs=101,
        logfile='./logs/lrs2_lm_100+20',
        try_restore_latest_checkpoint=True
    )

    experiment = avsr.LM(
        unit='character',
        unit_file='./avsr/misc/character_list',
        labels_train_record='/run/media/john_tukey/download/datasets/lrs2/tfrecords/characters_train_success.tfrecord',
        labels_test_record='/run/media/john_tukey/download/datasets/lrs2/tfrecords/characters_test_success.tfrecord',
        clip_gradients=True,
        decoder_units_per_layer=(256,),
        learning_rate=0.0001,
    )

    experiment.train(
        num_epochs=21,
        logfile='./logs/lrs2_lm_100+20',
        try_restore_latest_checkpoint=True
    )


if __name__ == '__main__':
    main()

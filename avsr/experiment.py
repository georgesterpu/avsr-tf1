from avsr import AVSR
from os import path


def run_experiment(
        video_train_record=None,
        video_test_record=None,
        labels_train_record=None,
        labels_test_record=None,
        audio_train_records=None,
        audio_test_records=None,
        unit='character',
        unit_list_file='./avsr/misc/character_list',
        iterations=None,
        learning_rates=None,
        logfile='tmp_experiment',
        warmup_epochs=0,
        warmup_max_len=50,
        input_modality='audio',
        **kwargs):

    full_logfile = path.join('./logs', logfile)

    ## warmup on short sentences
    if warmup_epochs >= 1:
        experiment = AVSR(
            unit=unit,
            unit_file=unit_list_file,
            audio_train_record=audio_train_records[0] if input_modality != 'video' else None,
            audio_test_record=audio_test_records[0] if input_modality != 'video' else None,
            video_train_record=video_train_record,
            video_test_record=video_test_record,
            labels_train_record=labels_train_record,
            labels_test_record=labels_test_record,
            learning_rate=learning_rates[0][0],
            max_sentence_length=warmup_max_len,
            **kwargs
        )

        with open(full_logfile, 'a') as f:
            f.write('Warm up on short sentences up to {} tokens for {} epochs \n'
                    .format(warmup_max_len, warmup_epochs))

        experiment.train(
            logfile=full_logfile,
            num_epochs=warmup_epochs,
            try_restore_latest_checkpoint=True
        )

        with open(full_logfile, 'a') as f:
            f.write(5 * '=' + '\n')
    ##

    if input_modality == 'video':
        iters = iterations[0]
        lr = learning_rates[0]
        experiment = AVSR(
            unit=unit,
            unit_file=unit_list_file,
            video_train_record=video_train_record,
            video_test_record=video_test_record,
            labels_train_record=labels_train_record,
            labels_test_record=labels_test_record,
            learning_rate=lr[0],
            **kwargs
        )
        experiment.train(
            logfile=full_logfile,
            num_epochs=iters[0] + 1,
            try_restore_latest_checkpoint=True
        )
        with open(full_logfile, 'a') as f:
            f.write(5*'=' + '\n')
        del experiment

        experiment = AVSR(
            unit=unit,
            unit_file=unit_list_file,
            video_train_record=video_train_record,
            video_test_record=video_test_record,
            labels_train_record=labels_train_record,
            labels_test_record=labels_test_record,
            learning_rate=lr[1],
            **kwargs
        )
        experiment.train(
            logfile=full_logfile,
            num_epochs=iters[1] + 1,
            try_restore_latest_checkpoint=True
        )
        with open(full_logfile, 'a') as f:
            f.write(20*'=' + '\n')

    else:
        for lr, iters, audio_train, audio_test in zip(learning_rates, iterations, audio_train_records, audio_test_records):
            experiment = AVSR(
                unit=unit,
                unit_file=unit_list_file,
                audio_train_record=audio_train,
                audio_test_record=audio_test,
                video_train_record=video_train_record,
                video_test_record=video_test_record,
                labels_train_record=labels_train_record,
                labels_test_record=labels_test_record,
                learning_rate=lr[0],
                **kwargs
            )
            experiment.train(
                logfile=full_logfile,
                num_epochs=iters[0]+1,
                try_restore_latest_checkpoint=True
            )

            with open(full_logfile, 'a') as f:
                f.write(5*'=' + '\n')

            experiment = AVSR(
                unit=unit,
                unit_file=unit_list_file,
                audio_train_record=audio_train,
                audio_test_record=audio_test,
                video_train_record=video_train_record,
                video_test_record=video_test_record,
                labels_train_record=labels_train_record,
                labels_test_record=labels_test_record,
                learning_rate=lr[1],
                **kwargs
            )
            experiment.train(
                logfile=full_logfile,
                num_epochs=iters[1]+1,
                try_restore_latest_checkpoint=True
            )

            with open(full_logfile, 'a') as f:
                f.write(20*'=' + '\n')



def run_experiment_mixedsnrs(
        video_train_record=None,
        video_test_record=None,
        labels_train_record=None,
        labels_test_record=None,
        audio_train_record=None,
        audio_test_record=None,
        unit='character',
        unit_list_file='./avsr/misc/character_list',
        iterations=None,
        learning_rates=None,
        architecture='unimodal',
        logfile='tmp_experiment',
        **kwargs):

    if architecture == 'unimodal':
        video_processing = None
    else:
        video_processing = 'resnet_cnn'

    full_logfile = path.join('./logs', logfile)


    for lr, iters in zip(learning_rates, iterations):
        experiment = AVSR(
            unit=unit,
            unit_file=unit_list_file,
            audio_processing='features',
            audio_train_record=audio_train_record,
            audio_test_record=audio_test_record,
            video_processing=video_processing,
            video_train_record=video_train_record,
            video_test_record=video_test_record,
            labels_train_record=labels_train_record,
            labels_test_record=labels_test_record,
            architecture=architecture,
            learning_rate=lr[0],
            **kwargs
        )
        experiment.train(
            logfile=full_logfile,
            num_epochs=iters[0]+1,
            try_restore_latest_checkpoint=True
        )

        with open(full_logfile, 'a') as f:
            f.write(5*'=' + '\n')

        experiment = AVSR(
            unit=unit,
            unit_file=unit_list_file,
            audio_processing='features',
            audio_train_record=audio_train_record,
            audio_test_record=audio_test_record,
            video_processing=video_processing,
            video_train_record=video_train_record,
            video_test_record=video_test_record,
            labels_train_record=labels_train_record,
            labels_test_record=labels_test_record,
            architecture=architecture,
            learning_rate=lr[1],
            **kwargs
        )
        experiment.train(
            logfile=full_logfile,
            num_epochs=iters[1]+1,
            try_restore_latest_checkpoint=True
        )

        with open(full_logfile, 'a') as f:
            f.write(20*'=' + '\n')


from os import path
from natsort import natsorted

_current_path = path.abspath(path.dirname(__file__))

viseme_file = path.join(_current_path, './htkconfigs/allVisemes.mlf')
phoneme_file = path.join(_current_path, './htkconfigs/allPhonemes.mlf')
character_file = path.join(_current_path, './htkconfigs/allCharacters.mlf')

viseme_list = path.join(_current_path, './htkconfigs/viseme_list')
phoneme_list = path.join(_current_path, './htkconfigs/phoneme_list')
character_list = path.join(_current_path, './htkconfigs/character_list')


volunteers = ('01M', '02M', '03F', '04M', '05F', '06M', '07F', '08F', '09F', '10M',
              '11F', '12M', '13F', '14M', '15F', '16M', '17F', '18M', '19M', '20M',
              '21M', '22M', '23M', '24M', '25M', '26M',        '28M', '29M', '30F',
              '31F', '32F', '33F', '34M',        '36F', '37F', '38F', '39M', '40F',
              '41M', '42M', '43F', '44F', '45F', '46F', '47M', '48M', '49F', '50F',
              '51F', '52M',        '54M', '55F', '56M', '57M', '58F', '59F',)
# volunteers 27, 35, 53 excluded for their non-irish accent (e.g. spanish, british)

lipspeakers = ('Lipspkr1', 'Lipspkr2', 'Lipspkr3')

def request_files(dataset_dir, protocol=None, speaker_type=None, speaker_id=None, remove_sa=False):
    r"""Generates the train/test split according to predefined protocols.
    If no protocol is defined, the function attempts to find all the video files located at `dataset_dir`
    and return a random train/test split.
    Parameters
    ----------
    dataset_dir
    protocol : `str` or `None`, optional
        Can be ``speaker_dependent``, ``speaker_independent``, ``single volunteer``
    speaker_type : 'str' or `None`, optional
        Can be ``volunteer`` or ``lipspeaker``
    speaker_id : `str`, optional
        A three character string encoding the ID of a volunteer, .e.g. ``01M``
    remove_sa : `bool`, optional
        Flag to remove the two `sa` sentences from the file list
        These are spoken by each volunteer

    Returns
    -------

    """
    if protocol == 'speaker_independent':
        train, test = _preload_files_speaker_independent(dataset_dir, speaker_type)
    elif protocol == 'speaker_dependent':
        train, test = _preload_files_speaker_dependent(dataset_dir)
    elif protocol == 'single_volunteer':
        train, test = _preload_files_single_volunteer(dataset_dir, speaker_id)
    else:
        raise Exception('unknown protocol')

    if remove_sa is True:
        train = [file for file in train if 'sa1' not in file and 'sa2' not in file]
        test = [file for file in test if 'sa1' not in file and 'sa2' not in file]

    return natsorted(train), natsorted(test)


def _read_file_contents(file):
    with open(file, 'r') as ftr:
        contents = ftr.read().splitlines()
    return contents


def _preload_files_speaker_dependent(dataset_dir):
    r"""Speaker-dependent protocol
    Each speaker contributes with 67 training sentences
    and 31 testing sentences
    Parameters
    ----------
    dataset_dir

    Returns
    -------

    """

    train_script = path.join(_current_path, 'splits/speaker-dependent/train.scp')
    test_script = path.join(_current_path, 'splits/speaker-dependent/test.scp')

    train_files = [path.join(dataset_dir, line) for line in _read_file_contents(train_script)]
    test_files = [path.join(dataset_dir, line) for line in _read_file_contents(test_script)]

    return train_files, test_files


def _preload_files_speaker_independent(dataset_dir, speaker_type='volunteer'):
    r"""Speaker-independent protocol
    There are 39 volunteers used for training
    and 17 for testing
    The remaining three are excluded
    Parameters
    ----------
    dataset_dir

    Returns
    -------

    """
    if speaker_type == 'volunteer':
        train_script = path.join(_current_path, 'splits/speaker-independent/volunteers_train.scp')
        test_script = path.join(_current_path, 'splits/speaker-independent/volunteers_test.scp')
        # train_script = path.join(_current_path, 'splits/speaker-independent/50')
        # test_script = path.join(_current_path, 'splits/speaker-independent/6')
    elif speaker_type == 'lipspeaker':
        train_script = path.join(_current_path, 'splits/speaker-independent/lipspeakers_train.scp')
        test_script = path.join(_current_path, 'splits/speaker-independent/lipspeakers_test.scp')
    else:
        raise Exception('Unknown speaker_type')

    train_files = [path.join(dataset_dir, line) for line in _read_file_contents(train_script)]
    test_files = [path.join(dataset_dir, line) for line in _read_file_contents(test_script)]

    return train_files, test_files


def _preload_files_single_volunteer(dataset_dir, speaker_id):
    r"""Loads the file of a single volunteer, maintaining the same train/test split
    from the speaker dependent protocol
    Parameters
    ----------
    dataset_dir
    speaker_id

    Returns
    -------

    """

    train_script = path.join(_current_path, 'splits/speaker-dependent/train.scp')
    test_script = path.join(_current_path, 'splits/speaker-dependent/test.scp')

    train_files = [path.join(dataset_dir, line) for line in _read_file_contents(train_script) if speaker_id in line]
    test_files = [path.join(dataset_dir, line) for line in _read_file_contents(test_script) if speaker_id in line]

    return train_files, test_files


def read_sentence_labels(filename, unit='viseme'):
    r"""Finds the labels associated with a sentence
    in a .mlf label file
    Parameters
    ----------
    filename : `str`
    unit : `viseme` or `phoneme`
        The modeled speech unit

    Returns
    -------
    label_seq : `list`
        A list of label symbols
    """
    file = path.splitext(path.split(filename)[1])[0]

    if unit == 'viseme':
        transcript = viseme_file
    elif unit == 'phoneme':
        transcript = phoneme_file
    elif unit == 'character':
        transcript = character_file
    else:
        raise Exception('only `viseme`, `phoneme` and `character` unit transcriptions are supported')

    with open(transcript, 'r') as f:
        contents = f.read()

    # start = contents.find(file)
    # end = contents.find('.\n', start)
    # sentence_transcript = contents[start:end].splitlines()[1:]
    #
    # label_seq = [item.split()[-1] for item in sentence_transcript]
    # return label_seq
    return _get_transcript_from_buffer(contents, file)


def read_all_sentences_labels(filenames, unit='viseme'):
    r"""
    Multi-file version of `read_sentence_labels`
    which prevents reading the ground truth file multiple times
    Parameters
    ----------
    filenames
    unit

    Returns
    -------

    """

    if unit == 'viseme':
        transcript = viseme_file
    elif unit == 'phoneme':
        transcript = phoneme_file
    elif unit == 'character':
        transcript = character_file
    else:
        raise Exception('only `viseme`, `phoneme` and `character` unit transcriptions are supported')

    with open(transcript, 'r') as f:
        contents = f.read()

    labels = {}
    for filename in filenames:
        file = path.splitext(path.split(filename)[1])[0]
        labels[filename] = _get_transcript_from_buffer(contents, file)

    return labels


def _get_transcript_from_buffer(buffer, file):
    start = buffer.find(file)
    end = buffer.find('.\n', start)
    sentence_transcript = buffer[start:end].splitlines()[1:]

    label_seq = [item.split()[-1] for item in sentence_transcript]
    return label_seq

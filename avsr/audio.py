import tensorflow as tf
from tensorflow.contrib import signal
import numpy as np


def compute_stfts(tensors, hparams):

    frame_length_samples = int((hparams.sample_rate / 1000) * hparams.frame_length_msec)
    frame_step_samples = int((hparams.sample_rate / 1000) * hparams.frame_step_msec)

    stfts = signal.stft(
        signals=tensors,
        frame_length=frame_length_samples,
        frame_step=frame_step_samples,
    )

    return stfts


def compute_log_mel_spectrograms(stfts, hparams):
    # power_spectrograms = tf.real(stfts * tf.conj(stfts))
    magnitude_spectrograms = tf.abs(stfts)

    num_spectrogram_bins = magnitude_spectrograms.shape[-1].value

    linear_to_mel_weight_matrix = signal.linear_to_mel_weight_matrix(
        hparams.num_mel_bins, num_spectrogram_bins, hparams.sample_rate, hparams.mel_lower_edge_hz,
        hparams.mel_upper_edge_hz)

    mel_spectrograms = tf.tensordot(
        magnitude_spectrograms, linear_to_mel_weight_matrix, 1)

    # Note: Shape inference for `tf.tensordot` does not currently handle this case.
    mel_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

    log_offset = 1e-6
    log_mel_spectrograms = tf.log(mel_spectrograms + log_offset)

    return log_mel_spectrograms


def compute_mfccs(log_mel_spectrograms, hparams):
    num_mfccs = hparams.num_mfccs
    mfccs = tf.contrib.signal.mfccs_from_log_mel_spectrograms(
        log_mel_spectrograms)[..., :num_mfccs]

    return mfccs


def process_audio(tensors, hparams, need_logmel=False, need_mfcc=False):
    stfts = compute_stfts(tensors, hparams)

    feature = stfts
    if need_logmel is True:
        feature = compute_log_mel_spectrograms(stfts, hparams)

    if need_mfcc is True:
        feature = compute_mfccs(feature, hparams)

    return feature


def read_wav_file(file, sr=16000):
    r"""
    Loads wav files from disk and resamples to 22050 Hz
    The output is shaped as [timesteps, 1]
    Parameters
    ----------
    file:
    sr: desired sampling rate

    Returns
    -------

    """
    import librosa
    data, sr = librosa.load(file, sr)
    return np.expand_dims(data, axis=-1)

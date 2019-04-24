import numpy as np
from os import path
from .audio import read_wav_file
current_path = path.abspath(path.dirname(__file__))


def generate_random_vector(length):
    random_sig = np.random.normal(0, 1, length)
    return np.expand_dims(random_sig, -1)


def add_noise(orig_signal, noise_type, snr, sampling_rate):

    signal_length = len(orig_signal)
    sig_power = np.sum(np.abs(orig_signal)**2)/signal_length
    target_snr = 10**(snr/10)

    if noise_type == 'white':
        noise_data = generate_random_vector(signal_length)
    elif noise_type == 'cafe':
        noise_data = read_wav_file(path.join(current_path, 'noise_data', 'cafeteria_babble.wav'), sampling_rate)
        noise_data = random_segment(noise_data, signal_length)
        sig_power /= np.sum(np.abs(noise_data)**2)/signal_length
    elif noise_type == 'street':
        noise_data = read_wav_file(path.join(current_path, 'noise_data', 'street_noise_downtown.wav'), sampling_rate)
        noise_data = random_segment(noise_data, signal_length)
        sig_power /= np.sum(np.abs(noise_data) ** 2) / signal_length
    else:
        raise Exception('unknown noise type, did you mean `white` ?')

    noise_power = float(sig_power/target_snr)
    noise_variance = np.sqrt(noise_power)

    noise = noise_variance * noise_data

    return np.expand_dims(orig_signal, axis=-1) + noise


def add_noise_cached(orig_signal, noise_type, noise_data, snr):
    signal_length = len(orig_signal)
    sig_power = np.sum(np.abs(orig_signal) ** 2) / signal_length
    target_snr = 10 ** (snr / 10)

    if noise_type == 'wgn':
        noise_data = generate_random_vector(signal_length)
    elif noise_type == 'cafe':
        noise_data = random_segment(noise_data, signal_length)
        sig_power /= np.sum(np.abs(noise_data) ** 2) / signal_length
    elif noise_type == 'street':
        noise_data = random_segment(noise_data, signal_length)
        sig_power /= np.sum(np.abs(noise_data) ** 2) / signal_length
    else:
        raise Exception('unknown noise type, did you mean `wgn` ?')

    noise_power = float(sig_power / target_snr)
    noise_variance = np.sqrt(noise_power)

    noise = noise_variance * noise_data

    return orig_signal + noise


def random_segment(data, target_len):
    from random import randint

    start_limit = len(data) - target_len
    start = randint(0, start_limit)

    return data[start:start + target_len]


def cache_noise(noise_type, sampling_rate):

    if noise_type == 'wgn':
        noise_data = None
    elif noise_type == 'cafe':
        noise_data = read_wav_file(path.join(current_path, 'noise_data', 'cafeteria_babble.wav'), sampling_rate)
    elif noise_type == 'street':
        noise_data = read_wav_file(path.join(current_path, 'noise_data', 'street_noise_downtown.wav'), sampling_rate)
    else:
        raise Exception('unknown noise type, did you mean `white` ?')

    return noise_data

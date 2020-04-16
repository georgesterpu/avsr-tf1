import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt



def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def get_av_timestamps(sentence):
    audio_stamps = np.arange(sentence.shape[1]) * 30

    video_stamps = []
    video_len, audio_len = sentence.shape
    x = np.arange(video_len)
    for step in range(audio_len):
        slice = sentence[:, step]

        try:
            mean = np.argmax(slice)
            # sigma = np.std(slice)
            sigma = 0.5
            p0 = [1, mean, sigma]
            popt, pcov = curve_fit(gaussian, x, slice, p0=p0, maxfev=200000)
            x2 = np.linspace(0, video_len, video_len * 10)
            new_curve = gaussian(x2,*popt)
            argmax_id = np.argmax(new_curve)
        except:
            argmax_id = np.argmax(slice) * 10
        video_stamps.append(argmax_id / 10)

    video_stamps = np.array(video_stamps) * 40

    # alternatively
    # video_stamps = np.argmax(sentence, axis=0) * 40
    tau = video_stamps - audio_stamps
    tau = tau[2:]  # skip "noise"

    return audio_stamps[2:], tau


def get_at_timestamps(sentence):
    # audio_stamps = np.arange(sentence.shape[1]) * 30

    audio_stamps = []
    audio_len, char_len = sentence.shape
    x = np.arange(audio_len)
    for step in range(char_len):
        slice = sentence[:, step]

        try:
            mean = np.argmax(slice)
            # sigma = np.std(slice)
            sigma = 0.5
            p0 = [1, mean, sigma]
            popt, pcov = curve_fit(gaussian, x, slice, p0=p0, maxfev=200000)
            x2 = np.linspace(0, audio_len, audio_len * 10)
            new_curve = gaussian(x2, *popt)
            argmax_id = np.argmax(new_curve)
        except:
            argmax_id = np.argmax(slice) * 10
        audio_stamps.append(argmax_id / 10)

    audio_stamps = np.array(audio_stamps) * 30

    return audio_stamps


def write_fig(audio_stamps, tau, title, fname):
    fig = plt.Figure(constrained_layout=True, figsize=(6, 6), dpi=300)
    # plt.rcParams.update({'font.size': 16})
    plt.plot(audio_stamps, tau, marker='o', markerfacecolor='m')
    plt.axis('off')
    # plt.title(title)
    # plt.xlabel('time [ms]')
    # plt.ylabel('tau [ms]')
    # plt.xticks(np.arange(0, audio_stamps[-1], step=200), fontsize=9)
    plt.savefig(fname, bbox_inches='tight', pad_inches=0, dpi=300, transparent=True)

    plt.clf()


def write_txt(audio_stamps, tau, fname):
    with open(fname, 'w') as f:
        for (a, t) in zip(audio_stamps, tau):
            f.write(str(a/1000) + ' ' + str(t) + '\n')


def write_praat_intensity(audio_stamps, tau, fname):
    import math
    from string import Template

    header = Template("""File type = "ooTextFile"
    Object class = "IntensityTier"

    xmin = 0 
    xmax = ${XMAX} 
    points: size = ${SIZE}
    """)

    pt = Template("""points [$INDEX]:
        number = $STAMP
        value = $LAG
    """)

    final = ''
    for idx, (timestamp, lag) in enumerate(zip(audio_stamps, tau)):
        current_pt = pt.substitute(INDEX=idx + 1, STAMP=float(timestamp/1000), LAG=lag)
        final += current_pt

    final = header.substitute(SIZE=idx + 1, XMAX=math.ceil(float(timestamp/1000))) + final

    with open(fname, 'w') as f:
        f.write(final)




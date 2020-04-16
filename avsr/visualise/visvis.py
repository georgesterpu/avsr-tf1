import numpy as np
import imageio
import json
from os import path, makedirs


def write_frames(png_dir, frames):
    makedirs(png_dir, exist_ok=True)
    png_paths = []
    for idx, frame in enumerate(frames):
        frame_path = path.join(png_dir, str(idx)+'.png')
        uint8_frame = (frame+1)*128
        uint8_frame = uint8_frame.astype(np.uint8)
        imageio.imwrite(frame_path, uint8_frame)
        png_paths.append(frame_path)

    slim_paths = [p.split('/', 4)[-1] for p in png_paths]
    return slim_paths


def write_json(
        av_sentence,
        at_sentence,
        labels,
        spectrogram,
        png_paths,
        json_file
):
    write_me = {"audio-video": av_sentence.T.tolist(),
                "text-audio": at_sentence.T.tolist(),
                "labels": labels,
                "spectrogram": spectrogram.split('/', 4)[-1],
                "video-frames": png_paths}

    with open(json_file, 'w') as fp:
        json.dump(write_me, fp)
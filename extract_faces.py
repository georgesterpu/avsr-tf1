import subprocess as sp
from os import path, makedirs
import sys
from multiprocessing import Pool
from itertools import repeat


def main(argv):
    r'''
    Multithreaded face-alignment using OpenFace
    '''

    num_threads = int(argv[1]) if len(argv) > 1 else 1

    openface_bin = '/run/media/john_tukey/download/software/OpenFace/build/bin/FeatureExtraction'
    out_dir = '/run/media/john_tukey/download/datasets/tcdtimit/aligned_openface/'
    makedirs(out_dir, exist_ok=True)

    dataset_dir = '/run/media/john_tukey/download/datasets/adapt/work/tcdtimit/'
    train_list = './datasets/tcdtimit/splits/speaker-dependent/train.scp'
    test_list = './datasets/tcdtimit/splits/speaker-dependent/test.scp'

    train = get_files(train_list, dataset_dir)
    test = get_files(test_list, dataset_dir)

    label_map = dict()
    for file in train + test:
        label_map[file] = path.splitext(file.split('tcdtimit/')[-1])[0]

    with Pool(num_threads) as p:
        p.starmap(process_one_batch,
              zip(chunks(train+test, 1000), repeat(openface_bin), repeat(label_map), repeat(out_dir)))


def process_one_batch(file_batch, openface_bin, label_map, out_dir):
    #  OpenFace does not have a Python interface yet, so we run the binary from Python
    outfs = []

    for file in file_batch:
        outf = path.join(out_dir, label_map[file])
        outfs.append(outf)

    cmd = [openface_bin]
    for i in range(len(file_batch)):
        cmd.append('-f')
        cmd.append(file_batch[i])

    for i in range(len(file_batch)):
        cmd.append('-of')
        cmd.append(outfs[i])

    cmd.extend(['-q', '-simscale', '1.0', '-simalign', '-pose',
                '-wild', '-multi-view', '1'])

    sp.run(cmd, check=True)


def chunks(l, n):
    # https://chrisalbon.com/python/data_wrangling/break_list_into_chunks_of_equal_size/
    for i in range(0, len(l), n):
        yield l[i:i+n]


def get_files(file_list, dataset_dir):
    with open(file_list, 'r') as f:
        contents = f.read().splitlines()

    contents = [path.join(dataset_dir, line.split()[0]) for line in contents]
    return contents


if __name__ == '__main__':
    main(sys.argv)

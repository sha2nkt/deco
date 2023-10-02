# load split.json and make an npz with all the folders in the test split
import argparse
import json
import os
import glob
import numpy as np

BEHAVE_PATH = '/ps/project/datasets/BEHAVE/sequences/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=BEHAVE_PATH)
    parser.add_argument('--split_file', type=str, default='data/preprocess/behave_test/split.json')
    parser.add_argument('--out_file', type=str, default='data/dataset_extras/behave/behave_simple_test.npz')
    args = parser.parse_args()

    with open(args.split_file, 'r') as f:
        split = json.load(f)

    test_split = split['test']

    # structs we use
    imgnames_ = []

    data = {}
    for seq_name in test_split:
        print(seq_name)
        seq_dir = os.path.join(args.data_dir, seq_name)
        # get recusive images in the seq_dir folder
        images = glob.glob(os.path.join(seq_dir, '**/*color.jpg'), recursive=True)
        print(len(images))
        images.sort()
        imgnames_.extend(images)

    np.savez(args.out_file, imgname=imgnames_,)
    print('Saved to ', args.out_file)


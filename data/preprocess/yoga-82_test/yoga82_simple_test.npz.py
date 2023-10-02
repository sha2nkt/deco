# load split.json and make an npz with all the folders in the test split
import argparse
import json
import os
import glob
import numpy as np

Yoga_82_PATH = '/is/cluster/work/stripathi/pycharm_remote/yogi/data/Yoga-82/yoga_dataset_images'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=Yoga_82_PATH)
    parser.add_argument('--out_file', type=str, default='data/dataset_extras/yoga-82/yoga-82_simple_test_20each.npz')
    args = parser.parse_args()

    # structs we use
    imgnames_ = []

    # get all ffolder names in the data_dir
    folders = glob.glob(os.path.join(args.data_dir, '*'))
    folders.sort()
    for folder in folders:
        print(folder)
        # get all images in the folder
        images = glob.glob(os.path.join(folder, '*.jpg'), recursive=True)
        print(len(images))
        # only take random 50 images from each folder
        images = np.random.choice(images, 20, replace=False)
        imgnames_.extend(images)


    np.savez(args.out_file, imgname=imgnames_,)
    print('Saved to ', args.out_file)


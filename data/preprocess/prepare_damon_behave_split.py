import os.path as osp
import os
import shutil
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

objects = {
    "backpack": 24,
    "chair": 56,
    "keyboard": 66,
    "suitcase": 28
}

def copy_images_to_behave_format(in_img_dir, in_image_list, in_part_dir, in_seg_dir, out_dir):
    """
    Copy images from in_img_dir to out_dir
    :param in_img_dir: input directory containing images
    :param out_dir: output directory to copy images to
    :return:
    """
    # read image list
    with open(in_image_list, 'r') as fp:
        img_list_dict = json.load(fp)

    for k, v in img_list_dict.items():
        out_dir_object = osp.join(out_dir, k)
        os.makedirs(out_dir_object, exist_ok=True)
        # copy images to out_dir
        for img_name in tqdm(v, dynamic_ncols=True):
            input_image_path = osp.join(in_img_dir, img_name)
            input_part_path = osp.join(in_part_dir, img_name.replace('.jpg', '_0.png'))
            input_seg_path = osp.join(in_seg_dir, img_name.replace('.jpg', '.png'))
            if not osp.exists(input_part_path) or not osp.exists(input_image_path) or not osp.exists(input_seg_path):
                print(f'{input_image_path} or {input_part_path} or {input_seg_path} does not exist')
                continue
            out_dir_image = osp.join(out_dir_object, img_name)
            os.makedirs(out_dir_image, exist_ok=True)
            shutil.copy(input_image_path, osp.join(out_dir_image, 'k1.color.jpg'))

            # load body mask
            body_mask = Image.open(input_part_path)
            # convert all non-zero pixels to 255
            body_mask = np.array(body_mask)
            body_mask[body_mask > 0] = 255
            body_mask = Image.fromarray(body_mask)
            body_mask.save(osp.join(out_dir_image, 'k1.person_mask.png'))

            # load seg mask
            body_mask = Image.open(input_seg_path)
            # convert all non-object pixels to 255
            body_mask = np.array(body_mask)
            object_num = objects[k]
            body_mask[body_mask == object_num] = 255
            body_mask[body_mask != 255] = 0
            body_mask = Image.fromarray(body_mask)
            body_mask.save(osp.join(out_dir_image, 'k1.object_rend.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_img_dir', type=str, default='/ps/project/datasets/HOT/Contact_Data/images/training')
    parser.add_argument('--in_part_dir', type=str, default='/ps/scratch/ps_shared/stripathi/deco/4agniv/hot/parts/training')
    parser.add_argument('--in_seg_dir', type=str, default='/ps/scratch/ps_shared/stripathi/deco/4agniv/hot_behave_split/agniv/masks')
    parser.add_argument('--in_image_list', type=str, default='/ps/scratch/ps_shared/stripathi/deco/4agniv/hot_behave_split/imgnames_per_object_dict.json')
    parser.add_argument('--out_dir', type=str, default='/ps/scratch/ps_shared/stripathi/deco/4agniv/hot_behave_split/training')
    args = parser.parse_args()
    copy_images_to_behave_format(args.in_img_dir, args.in_image_list, args.in_part_dir, args.in_seg_dir, args.out_dir)
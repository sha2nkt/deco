import os
os.environ["CDF_LIB"] = "/is/cluster/scratch/stripathi/data/cdf37_1-dist/src/lib"

import cv2
import pandas as pd
import json
import glob
import h5py
import torch
import trimesh
import numpy as np
import pickle as pkl
from xml.dom import minidom
import xml.etree.ElementTree as ET
from tqdm import tqdm
from spacepy import pycdf
# from .read_openpose import read_openpose
import sys
sys.path.append('../../')
from models import hmr, SMPL
import config
import constants

import shutil

import smplx
import pytorch3d.transforms as p3dt

from utils.geometry import batch_rodrigues, batch_rot2aa, ea2rm


model_type = 'smplx'
model_folder = '/ps/project/common/smplifyx/models/'
body_model_params = dict(model_path=model_folder,
                         model_type=model_type,
                         create_global_orient=True,
                         create_body_pose=True,
                         create_betas=True,
                         num_betas=10,
                         create_left_hand_pose=True,
                         create_right_hand_pose=True,
                         create_expression=True,
                         create_jaw_pose=True,
                         create_leye_pose=True,
                         create_reye_pose=True,
                         create_transl=True,
                         use_pca=False)
body_model = smplx.create(gender='neutral', **body_model_params).to('cuda')

def rich_extract(img_dataset_path, out_path, split=None, vis_path=None, visualize=False, downsample_factor=4):

    # structs we use
    imgnames_ = []
    poses_, shapes_, transls_ = [], [], []
    cams_k_ = []
    contact_label_ = []
    scene_seg_, part_seg_ = [], []
    
    for i, fl in tqdm(enumerate(sorted(os.listdir(os.path.join(img_dataset_path, 'images', split)))), dynamic_ncols=True):
        ind = fl.index('cam')
        location = fl[:ind-1]

        cam_num = fl[ind:ind+6]

        img = fl[ind+7:-3] + 'jpeg'

        imgname = os.path.join(location, cam_num, img)

        mask_name = fl
        sp = mask_name.split('_')
        indx = mask_name.index('cam')
        st = mask_name[indx-1:indx+7]
        mask_name = mask_name.replace(st, '/')
        mask_name = mask_name[:-7]
        new_p = mask_name.split('/')
        mask_name = new_p[0] + '/' + new_p[1] + '/' + sp[1] + '.pkl'
        mask_path = os.path.join(img_dataset_path, 'labels', split, mask_name)
        df = pd.read_pickle(mask_path)
        mask = df['contact'] 

        scene_path = os.path.join(img_dataset_path, 'segmentation_masks', split, fl[:-3] + 'png')

        part_path = os.path.join(img_dataset_path, 'parts', split, fl[:-3] + 'png')

        dataset_path = '/ps/project/datasets/RICH'

        ind = fl.index('cam')
        frame_id = fl[:ind-1]
        location = frame_id.split('_')[0]

        if location == 'LectureHall':
            if 'chair' in frame_id:
                cam2world_location = location + '_' + 'chair'
            else:
                cam2world_location = location + '_' + 'yoga'  
        else:
            cam2world_location = location        

        img_num = fl.split('_')[-2]

        cam_num = int(fl.split('_')[-1][:2])

        # get ioi2scan transformation per sequence
        ioi2scan_fn = os.path.join(dataset_path, 'website_release/multicam2world', cam2world_location + '_multicam2world.json')

        try:
            camera_fn = os.path.join(dataset_path, 'rich_toolkit/data/scan_calibration', location, f'calibration/{cam_num:03d}.xml')
            focal_length_x, focal_length_y, camC, camR, camT, _, _, _ = extract_cam_param_xml(camera_fn)
        except:
            print(f'camera calibration file not found: {camera_fn}')
            continue

        # path to smpl params
        smplx_param = os.path.join(dataset_path, 'rich_toolkit/data/bodies', split, frame_id, str(img_num), frame_id.split('_')[1] + '.pkl')

        # get smpl parameters
        ## body resides in multi-ioi coordidate, where camera 0 is world zero.
        with open(smplx_param, 'rb') as f:
            body_params = pkl.load(f)
            # in ioi coordinates: cam 0
            beta = body_params['betas']
            pose_aa = body_params['body_pose']
            pose_rotmat = p3dt.axis_angle_to_matrix(torch.FloatTensor(pose_aa.reshape(-1,3))).numpy()

            transl = body_params['transl']
            global_orient = body_params['global_orient']
            global_orient = p3dt.axis_angle_to_matrix(torch.FloatTensor(global_orient.reshape(-1,3))).numpy()

        smpl_body_cam0 = body_model(betas=torch.FloatTensor(beta).to('cuda')) # canonical body with shape
        vertices_cam0 = smpl_body_cam0.vertices.detach().cpu().numpy().squeeze()
        joints_cam0 = smpl_body_cam0.joints.detach().cpu().numpy()
        pelvis_cam0 = joints_cam0[:, 0, :]

        # ## rigid transformation between multi-ioi and Leica scan (world)
        with open(ioi2scan_fn, 'r') as f:
            ioi2scan_dict = json.load(f)
            R_ioi2world = np.array(ioi2scan_dict['R']) # Note: R is transposed
            t_ioi2world= np.array(ioi2scan_dict['t']).reshape(1, 3)     

        # # get SMPL params in camera coordinates
        global_orient_cam = np.matmul(np.array(camR), global_orient)
        full_pose_rotmat_cam = np.concatenate((global_orient_cam, pose_rotmat), axis=0).squeeze()
        theta_cam = batch_rot2aa(torch.FloatTensor(full_pose_rotmat_cam)).reshape(-1, 66).cpu().numpy()

        # read GT 2D keypoints
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = focal_length_x / downsample_factor
        K[1, 1] = focal_length_y / downsample_factor
        K[:2, 2:] = camC.T / downsample_factor

        # get camera parameters wrt to scan
        R_worldtocam = np.matmul(camR, R_ioi2world) # Note: R_ioi2world is transposed
        T_worldtocam = -t_ioi2world + camT

        # store data
        imgnames_.append(os.path.join('/ps/project/datasets/RICH_JPG', split, imgname))
        contact_label_.append(mask)
        scene_seg_.append(scene_path)
        part_seg_.append(part_path)
        poses_.append(theta_cam.squeeze())
        transls_.append(transl.squeeze())
        shapes_.append(beta.squeeze())
        cams_k_.append(K.tolist())

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, f'rich_{split}_smplx.npz')
    np.savez(out_file, imgname=imgnames_,
                       pose=poses_,
                       transl=transls_,
                       shape=shapes_,
                       cam_k=cams_k_,
                       contact_label=contact_label_,
                        scene_seg=scene_seg_,
                        part_seg=part_seg_
             )
    print('Saved to ', out_file)

def rectify_pose(camera_r, body_aa):
    body_r = batch_rodrigues(body_aa).reshape(-1,3,3)
    final_r = camera_r @ body_r
    body_aa = batch_rot2aa(final_r)
    return body_aa


def extract_cam_param_xml(xml_path: str = '', dtype=float):
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)

    extrinsics_mat = [float(s) for s in tree.find('./CameraMatrix/data').text.split()]
    intrinsics_mat = [float(s) for s in tree.find('./Intrinsics/data').text.split()]
    distortion_vec = [float(s) for s in tree.find('./Distortion/data').text.split()]

    focal_length_x = intrinsics_mat[0]
    focal_length_y = intrinsics_mat[4]
    center = np.array([[intrinsics_mat[2], intrinsics_mat[5]]], dtype=dtype)

    rotation = np.array([[extrinsics_mat[0], extrinsics_mat[1], extrinsics_mat[2]],
                         [extrinsics_mat[4], extrinsics_mat[5], extrinsics_mat[6]],
                         [extrinsics_mat[8], extrinsics_mat[9], extrinsics_mat[10]]], dtype=dtype)

    translation = np.array([[extrinsics_mat[3], extrinsics_mat[7], extrinsics_mat[11]]], dtype=dtype)

    # t = -Rc --> c = -R^Tt
    cam_center = [-extrinsics_mat[0] * extrinsics_mat[3] - extrinsics_mat[4] * extrinsics_mat[7] - extrinsics_mat[8] *
                  extrinsics_mat[11],
                  -extrinsics_mat[1] * extrinsics_mat[3] - extrinsics_mat[5] * extrinsics_mat[7] - extrinsics_mat[9] *
                  extrinsics_mat[11],
                  -extrinsics_mat[2] * extrinsics_mat[3] - extrinsics_mat[6] * extrinsics_mat[7] - extrinsics_mat[10] *
                  extrinsics_mat[11]]

    cam_center = np.array([cam_center], dtype=dtype)

    k1 = np.array([distortion_vec[0]], dtype=dtype)
    k2 = np.array([distortion_vec[1]], dtype=dtype)

    return focal_length_x, focal_length_y, center, rotation, translation, cam_center, k1, k2

rich_extract(img_dataset_path='/is/cluster/work/achatterjee/rich', out_path='/is/cluster/work/achatterjee/rich/npzs', split='train')
rich_extract(img_dataset_path='/is/cluster/work/achatterjee/rich', out_path='/is/cluster/work/achatterjee/rich/npzs', split='val')
rich_extract(img_dataset_path='/is/cluster/work/achatterjee/rich', out_path='/is/cluster/work/achatterjee/rich/npzs', split='test')

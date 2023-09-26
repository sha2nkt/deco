import cv2
import os
import trimesh
import PIL.Image as pil_img
import numpy as np
import pyrender
from common import constants

os.environ['PYOPENGL_PLATFORM'] = 'egl'

def render_image(scene, img_res, img=None, viewer=False):
    '''
    Render the given pyrender scene and return the image. Can also overlay the mesh on an image.
    '''
    if viewer:
        pyrender.Viewer(scene, use_raymond_lighting=True)
        return 0
    else:
        r = pyrender.OffscreenRenderer(viewport_width=img_res,
                                       viewport_height=img_res,
                                       point_size=1.0)
        color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0

        if img is not None:
            valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
            input_img = img.detach().cpu().numpy()
            output_img = (color[:, :, :-1] * valid_mask +
                          (1 - valid_mask) * input_img)
        else:
            output_img = color
        return output_img

def create_scene(mesh, img, focal_length=500, camera_center=250, img_res=500):
    # Setup the scene
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0],
                           ambient_light=(0.3, 0.3, 0.3))
    # add mesh for camera
    camera_pose = np.eye(4)
    camera_rotation = np.eye(3, 3)
    camera_translation = np.array([0., 0, 2.5])
    camera_pose[:3, :3] = camera_rotation
    camera_pose[:3, 3] = camera_rotation @ camera_translation
    pyrencamera = pyrender.camera.IntrinsicsCamera(
        fx=focal_length, fy=focal_length,
        cx=camera_center, cy=camera_center)
    scene.add(pyrencamera, pose=camera_pose)
    # create and add light
    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1)
    light_pose = np.eye(4)
    for lp in [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1]]:
        light_pose[:3, 3] = mesh.vertices.mean(0) + np.array(lp)
        # out_mesh.vertices.mean(0) + np.array(lp)
        scene.add(light, pose=light_pose)
    # add body mesh
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    mesh_images = []

    # resize input image to fit the mesh image height
    # print(img.shape)
    img_height = img_res
    img_width = int(img_height * img.shape[1] / img.shape[0])
    img = cv2.resize(img, (img_width, img_height))
    mesh_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    for sideview_angle in [0, 90, 180, 270]:
        out_mesh = mesh.copy()
        rot = trimesh.transformations.rotation_matrix(
            np.radians(sideview_angle), [0, 1, 0])
        out_mesh.apply_transform(rot)
        out_mesh = pyrender.Mesh.from_trimesh(
            out_mesh,
            material=material)
        mesh_pose = np.eye(4)
        scene.add(out_mesh, pose=mesh_pose, name='mesh')
        output_img = render_image(scene, img_res)
        output_img = pil_img.fromarray((output_img * 255).astype(np.uint8))
        output_img = np.asarray(output_img)[:, :, :3]
        mesh_images.append(output_img)
        # delete the previous mesh
        prev_mesh = scene.get_nodes(name='mesh').pop()
        scene.remove_node(prev_mesh)

    # show upside down view
    for topview_angle in [90, 270]:
        out_mesh = mesh.copy()
        rot = trimesh.transformations.rotation_matrix(
            np.radians(topview_angle), [1, 0, 0])
        out_mesh.apply_transform(rot)
        out_mesh = pyrender.Mesh.from_trimesh(
            out_mesh,
            material=material)
        mesh_pose = np.eye(4)
        scene.add(out_mesh, pose=mesh_pose, name='mesh')
        output_img = render_image(scene, img_res)
        output_img = pil_img.fromarray((output_img * 255).astype(np.uint8))
        output_img = np.asarray(output_img)[:, :, :3]
        mesh_images.append(output_img)
        # delete the previous mesh
        prev_mesh = scene.get_nodes(name='mesh').pop()
        scene.remove_node(prev_mesh)

    # stack images
    IMG = np.hstack(mesh_images)
    IMG = pil_img.fromarray(IMG)
    IMG.thumbnail((3000, 3000))
    return IMG

# img = cv2.imread('../samples/prox_N3OpenArea_03301_01_s001_frame_00694.jpg')
# mesh = trimesh.load('../samples/mesh.ply', process=False)
# comb_img = create_scene(mesh, img)
# comb_img.save('../samples/combined_image.png')

def unsplit(img, palette):
    rgb_img = np.zeros((img.shape[0], img.shape[1], 3))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            id = np.argmax(img[i, j, :])
            rgb_img[i, j, :] = palette[id]

    return rgb_img

def gen_render(output, normalize=True):
    img = output['img'].cpu().numpy()
    contact_labels_3d = output['contact_labels_3d_gt'].cpu().numpy()
    contact_labels_3d_pred = output['contact_labels_3d_pred'].cpu().numpy()
    sem_mask_gt = output['sem_mask_gt'].cpu().numpy()
    sem_mask_pred = output['sem_mask_pred'].cpu().numpy()
    part_mask_gt = output['part_mask_gt'].cpu().numpy()
    part_mask_pred = output['part_mask_pred'].cpu().numpy()
    contact_2d_gt_rgb = output['contact_2d_gt'].cpu().numpy()
    contact_2d_pred_rgb = output['contact_2d_pred_rgb'].cpu().numpy()

    mesh_path = './data/smpl/smpl_neutral_tpose.ply'
    gt_mesh = trimesh.load(mesh_path, process=False)
    pred_mesh = trimesh.load(mesh_path, process=False)

    img = np.transpose(img[0], (1, 2, 0))
    if normalize:
        # unnormalize the image before displaying
        mean = np.array(constants.IMG_NORM_MEAN, dtype=np.float32)
        std = np.array(constants.IMG_NORM_STD, dtype=np.float32)
        img = img * std + mean
    img = img * 255
    img = img.astype(np.uint8)
    color = np.array([0, 0, 0, 255])
    th = 0.5

    contact_labels_3d = contact_labels_3d[0, :]
    for vid, val in enumerate(contact_labels_3d):
        if val >= th:
            gt_mesh.visual.vertex_colors[vid] = color

    contact_labels_3d_pred = contact_labels_3d_pred[0, :]
    for vid, val in enumerate(contact_labels_3d_pred):
        if val >= th:
            pred_mesh.visual.vertex_colors[vid] = color

    gt_rend = create_scene(gt_mesh, img)
    pred_rend = create_scene(pred_mesh, img)

    sem_palette = [[220, 20, 60], [119, 11, 32], [0, 0, 142], [0, 0, 230], [106, 0, 228], [0, 60, 100], [0, 80, 100], [0, 0, 70], [0, 0, 192], [250, 170, 30], [100, 170, 30], [220, 220, 0], [175, 116, 175], [250, 0, 30], [165, 42, 42], [255, 77, 255], [0, 226, 252], [182, 182, 255], [0, 82, 0], [120, 166, 157], [110, 76, 0], [174, 57, 255], [199, 100, 0], [72, 0, 118], [255, 179, 240], [0, 125, 92], [209, 0, 151], [188, 208, 182], [0, 220, 176], [255, 99, 164], [92, 0, 73], [133, 129, 255], [78, 180, 255], [0, 228, 0], [174, 255, 243], [45, 89, 255], [134, 134, 103], [145, 148, 174], [255, 208, 186], [197, 226, 255], [171, 134, 1], [109, 63, 54], [207, 138, 255], [151, 0, 95], [9, 80, 61], [84, 105, 51], [74, 65, 105], [166, 196, 102], [208, 195, 210], [255, 109, 65], [0, 143, 149], [179, 0, 194], [209, 99, 106], [5, 121, 0], [227, 255, 205], [147, 186, 208], [153, 69, 1], [3, 95, 161], [163, 255, 0], [119, 0, 170], [0, 182, 199], [0, 165, 120], [183, 130, 88], [95, 32, 0], [130, 114, 135], [110, 129, 133], [166, 74, 118], [219, 142, 185], [79, 210, 114], [178, 90, 62], [65, 70, 15], [127, 167, 115], [59, 105, 106], [142, 108, 45], [196, 172, 0], [95, 54, 80], [128, 76, 255], [201, 57, 1], [246, 0, 122], [191, 162, 208], [255, 255, 128], [147, 211, 203], [150, 100, 100], [168, 171, 172], [146, 112, 198], [210, 170, 100], [92, 136, 89], [218, 88, 184], [241, 129, 0], [217, 17, 255], [124, 74, 181], [70, 70, 70], [255, 228, 255], [154, 208, 0], [193, 0, 92], [76, 91, 113], [255, 180, 195], [106, 154, 176], [230, 150, 140], [60, 143, 255], [128, 64, 128], [92, 82, 55], [254, 212, 124], [73, 77, 174], [255, 160, 98], [255, 255, 255], [104, 84, 109], [169, 164, 131], [225, 199, 255], [137, 54, 74], [135, 158, 223], [7, 246, 231], [107, 255, 200], [58, 41, 149], [183, 121, 142], [255, 73, 97], [107, 142, 35], [190, 153, 153], [146, 139, 141], [70, 130, 180], [134, 199, 156], [209, 226, 140], [96, 36, 108], [96, 96, 96], [64, 170, 64], [152, 251, 152], [208, 229, 228], [206, 186, 171], [152, 161, 64], [116, 112, 0], [0, 114, 143], [102, 102, 156], [250, 141, 255]]
    # part_palette = [(0,0,0), (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0), (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0), (52,86,128), (0,128,0), (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (255,255,0), (255,170,0)]
    part_palette = [[0, 0, 0], [220, 20, 60], [119, 11, 32], [0, 0, 142], [0, 0, 230], [106, 0, 228], [0, 60, 100], [0, 80, 100], [0, 0, 70], [0, 0, 192], [250, 170, 30], [100, 170, 30], [220, 220, 0], [175, 116, 175], [250, 0, 30], [165, 42, 42], [255, 77, 255], [0, 226, 252], [182, 182, 255], [0, 82, 0], [120, 166, 157], [110, 76, 0], [174, 57, 255], [199, 100, 0], [72, 0, 118], [255, 179, 240]]
    hot_palette = [[0, 0, 0], [220, 20, 60], [119, 11, 32], [0, 0, 142], [0, 0, 230], [106, 0, 228], [0, 60, 100], [0, 80, 100], [0, 0, 70], [0, 0, 192], [250, 170, 30], [100, 170, 30], [220, 220, 0], [175, 116, 175], [250, 0, 30], [165, 42, 42], [255, 77, 255], [0, 226, 252]]

    sem_mask_gt = np.transpose(sem_mask_gt[0], (1, 2, 0))*255
    sem_mask_gt = sem_mask_gt.astype(np.uint8)
    sem_mask_pred = np.transpose(sem_mask_pred[0], (1, 2, 0))*255
    sem_mask_pred = sem_mask_pred.astype(np.uint8)
    part_mask_gt = np.transpose(part_mask_gt[0], (1, 2, 0))*255
    part_mask_gt = part_mask_gt.astype(np.uint8)
    part_mask_pred = np.transpose(part_mask_pred[0], (1, 2, 0))*255
    part_mask_pred = part_mask_pred.astype(np.uint8)
    contact_2d_gt_rgb = contact_2d_gt_rgb[0]*255
    contact_2d_gt_rgb = contact_2d_gt_rgb.astype(np.uint8)
    contact_2d_pred_rgb = contact_2d_pred_rgb[0]*255
    contact_2d_pred_rgb = contact_2d_pred_rgb.astype(np.uint8)

    sem_mask_rgb = unsplit(sem_mask_gt, sem_palette)
    sem_pred_rgb = unsplit(sem_mask_pred, sem_palette)
    part_mask_rgb = unsplit(part_mask_gt, part_palette)
    part_pred_rgb = unsplit(part_mask_pred, part_palette)

    sem_mask_rgb = sem_mask_rgb.astype(np.uint8)
    sem_pred_rgb = sem_pred_rgb.astype(np.uint8)
    part_mask_rgb = part_mask_rgb.astype(np.uint8)
    part_pred_rgb = part_pred_rgb.astype(np.uint8)

    sem_mask_rgb = pil_img.fromarray(sem_mask_rgb)
    sem_pred_rgb = pil_img.fromarray(sem_pred_rgb)
    part_mask_rgb = pil_img.fromarray(part_mask_rgb)
    part_pred_rgb = pil_img.fromarray(part_pred_rgb)
    contact_2d_gt_rgb = pil_img.fromarray(contact_2d_gt_rgb)
    contact_2d_pred_rgb = pil_img.fromarray(contact_2d_pred_rgb)

    tot_rend = pil_img.new('RGB', (3000, 2000))
    tot_rend.paste(gt_rend, (0, 0))
    tot_rend.paste(pred_rend, (0, 450))
    tot_rend.paste(sem_mask_rgb, (0, 900))
    tot_rend.paste(sem_pred_rgb, (400, 900))
    tot_rend.paste(part_mask_rgb, (0, 1300))
    tot_rend.paste(part_pred_rgb, (400, 1300))
    tot_rend.paste(contact_2d_gt_rgb, (0, 1700))
    tot_rend.paste(contact_2d_pred_rgb, (400, 1700))
    return tot_rend
import cv2
import os
import trimesh
import PIL.Image as pil_img
import numpy as np
import pyrender

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

def gen_render(img, pred):
  mesh_path = '/is/cluster/work/achatterjee/dca_contact/data/smpl/smpl_neutral_tpose.ply'
  pred_mesh = trimesh.load(mesh_path, process=False)

  img = np.transpose(img[0], (1, 2, 0))*255
  img = img.astype(np.uint8)
  color = np.array([0, 0, 0, 255])
  th = 0.5
  
  cont = pred[0, 0, :]
  for vid, val in enumerate(cont):
    if val >= th:
      pred_mesh.visual.vertex_colors[vid] = color

  pred_rend = create_scene(pred_mesh, img)

  return pred_rend
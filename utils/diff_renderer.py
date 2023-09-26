# from https://gitlab.tuebingen.mpg.de/mkocabas/projects/-/blob/master/pare/pare/utils/diff_renderer.py

import torch
import numpy as np
import torch.nn as nn

from pytorch3d.renderer import (
        PerspectiveCameras,
        RasterizationSettings,
        DirectionalLights,
        BlendParams,
        HardFlatShader,
        MeshRasterizer,
        TexturesVertex,
        TexturesAtlas
    )
from pytorch3d.structures import Meshes

from .image_utils import get_default_camera
from .smpl_uv import get_tenet_texture


class MeshRendererWithDepth(nn.Module):
    """
    A class for rendering a batch of heterogeneous meshes. The class should
    be initialized with a rasterizer and shader class which each have a forward
    function.
    """

    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        """
        Render a batch of images from a batch of meshes by rasterizing and then
        shading.

        NOTE: If the blur radius for rasterization is > 0.0, some pixels can
        have one or more barycentric coordinates lying outside the range [0, 1].
        For a pixel with out of bounds barycentric coordinates with respect to a
        face f, clipping is required before interpolating the texture uv
        coordinates and z buffer so that the colors and depths are limited to
        the range for the corresponding face.
        """
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)

        mask = (fragments.zbuf > -1).float()

        zbuf = fragments.zbuf.view(images.shape[0], -1)
        # print(images.shape, zbuf.shape)
        depth = (zbuf - zbuf.min(-1, keepdims=True).values) / \
                (zbuf.max(-1, keepdims=True).values - zbuf.min(-1, keepdims=True).values)
        depth = depth.reshape(*images.shape[:3] + (1,))

        images = torch.cat([images[:, :, :, :3], mask, depth], dim=-1)
        return images


class DifferentiableRenderer(nn.Module):
    def __init__(
            self,
            img_h,
            img_w,
            focal_length,
            device='cuda',
            background_color=(0.0, 0.0, 0.0),
            texture_mode='smplpix',
            vertex_colors=None,
            face_textures=None,
            smpl_faces=None,
            is_train=False,
            is_cam_batch=False,
    ):
        super(DifferentiableRenderer, self).__init__()
        self.x = 'a'
        self.img_h = img_h
        self.img_w = img_w
        self.device = device
        self.focal_length = focal_length
        K, R = get_default_camera(focal_length, img_h, img_w, is_cam_batch=is_cam_batch)
        K, R = K.to(device), R.to(device)

        # T = torch.tensor([[0, 0, 2.5 * self.focal_length / max(self.img_h, self.img_w)]]).to(device)
        if is_cam_batch:
            T = torch.zeros((K.shape[0], 3)).to(device)
        else:
            T = torch.tensor([[0.0, 0.0, 0.0]]).to(device)
        self.background_color = background_color
        self.renderer = None
        smpl_faces = smpl_faces

        if texture_mode == 'smplpix':
            face_colors = get_tenet_texture(mode=texture_mode).to(device).float()
            vertex_colors = torch.from_numpy(
                np.load(f'data/smpl/{texture_mode}_vertex_colors.npy')[:,:3]
            ).unsqueeze(0).to(device).float()
        if texture_mode == 'partseg':
            vertex_colors = vertex_colors[..., :3].unsqueeze(0).to(device)
            face_colors = face_textures.to(device)
        if texture_mode == 'deco':
            vertex_colors = vertex_colors[..., :3].to(device)
            face_colors = face_textures.to(device)

        self.register_buffer('K', K)
        self.register_buffer('R', R)
        self.register_buffer('T', T)
        self.register_buffer('face_colors', face_colors)
        self.register_buffer('vertex_colors', vertex_colors)
        self.register_buffer('smpl_faces', smpl_faces)

        self.set_requires_grad(is_train)

    def set_requires_grad(self, val=False):
        self.K.requires_grad_(val)
        self.R.requires_grad_(val)
        self.T.requires_grad_(val)
        self.face_colors.requires_grad_(val)
        self.vertex_colors.requires_grad_(val)
        # check if smpl_faces is a FloatTensor as requires_grad_ is not defined for LongTensor
        if isinstance(self.smpl_faces, torch.FloatTensor):
            self.smpl_faces.requires_grad_(val)

    def forward(self, vertices, faces=None, R=None, T=None):
        raise NotImplementedError


class Pytorch3D(DifferentiableRenderer):
    def __init__(
            self,
            img_h,
            img_w,
            focal_length,
            device='cuda',
            background_color=(0.0, 0.0, 0.0),
            texture_mode='smplpix',
            vertex_colors=None,
            face_textures=None,
            smpl_faces=None,
            model_type='smpl',
            is_train=False,
            is_cam_batch=False,
    ):
        super(Pytorch3D, self).__init__(
            img_h,
            img_w,
            focal_length,
            device=device,
            background_color=background_color,
            texture_mode=texture_mode,
            vertex_colors=vertex_colors,
            face_textures=face_textures,
            smpl_faces=smpl_faces,
            is_train=is_train,
            is_cam_batch=is_cam_batch,
        )

        # this R converts the camera from pyrender NDC to
        # OpenGL coordinate frame. It is basicall R(180, X) x R(180, Y)
        # I manually defined it here for convenience
        self.R = self.R @ torch.tensor(
            [[[ -1.0,  0.0, 0.0],
              [  0.0, -1.0, 0.0],
              [  0.0,  0.0, 1.0]]],
            dtype=self.R.dtype, device=self.R.device,
        )

        if is_cam_batch:
            focal_length = self.focal_length
        else:
            focal_length = self.focal_length[None, :]

        principal_point = ((self.img_w // 2, self.img_h // 2),)
        image_size = ((self.img_h, self.img_w),)

        cameras = PerspectiveCameras(
            device=self.device,
            focal_length=focal_length,
            principal_point=principal_point,
            R=self.R,
            T=self.T,
            in_ndc=False,
            image_size=image_size,
        )

        for param in cameras.parameters():
            param.requires_grad_(False)

        raster_settings = RasterizationSettings(
            image_size=(self.img_h, self.img_w),
            blur_radius=0.0,
            max_faces_per_bin=20000,
            faces_per_pixel=1,
        )

        lights = DirectionalLights(
            device=self.device,
            ambient_color=((1.0, 1.0, 1.0),),
            diffuse_color=((0.0, 0.0, 0.0),),
            specular_color=((0.0, 0.0, 0.0),),
            direction=((0, 1, 0),),
        )

        blend_params = BlendParams(background_color=self.background_color)

        shader = HardFlatShader(device=self.device,
                                cameras=cameras,
                                blend_params=blend_params,
                                lights=lights)

        self.textures = TexturesVertex(verts_features=self.vertex_colors)

        self.renderer = MeshRendererWithDepth(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=shader,
        )

    def forward(self, vertices, faces=None, R=None, T=None, face_atlas=None):
        batch_size = vertices.shape[0]
        if faces is None:
            faces = self.smpl_faces.expand(batch_size, -1, -1)

        if R is None:
            R = self.R.expand(batch_size, -1, -1)

        if T is None:
            T = self.T.expand(batch_size, -1)

        # convert camera translation to pytorch3d coordinate frame
        T = torch.bmm(R, T.unsqueeze(-1)).squeeze(-1)

        vertex_textures = TexturesVertex(
            verts_features=self.vertex_colors.expand(batch_size, -1, -1)
        )

        # face_textures needed because vertex_texture cause interpolation at boundaries
        if face_atlas:
            face_textures = TexturesAtlas(atlas=face_atlas)
        else:
            face_textures = TexturesAtlas(atlas=self.face_colors)

        # we may need to rotate the mesh
        meshes = Meshes(verts=vertices, faces=faces, textures=face_textures)
        images = self.renderer(meshes, R=R, T=T)
        images = images.permute(0, 3, 1, 2)
        return images


class NeuralMeshRenderer(DifferentiableRenderer):
    def __init__(self, *args, **kwargs):
        import neural_renderer as nr

        super(NeuralMeshRenderer, self).__init__(*args, **kwargs)

        self.neural_renderer = nr.Renderer(
            dist_coeffs=None,
            orig_size=self.img_size,
            image_size=self.img_size,
            light_intensity_ambient=1,
            light_intensity_directional=0,
            anti_aliasing=False,
        )

    def forward(self, vertices, faces=None, R=None, T=None):
        batch_size = vertices.shape[0]
        if faces is None:
            faces = self.smpl_faces.expand(batch_size, -1, -1)

        if R is None:
            R = self.R.expand(batch_size, -1, -1)

        if T is None:
            T = self.T.expand(batch_size, -1)
        rgb, depth, mask = self.neural_renderer(
            vertices,
            faces,
            textures=self.face_colors.expand(batch_size, -1, -1, -1, -1, -1),
            K=self.K.expand(batch_size, -1, -1),
            R=R,
            t=T.unsqueeze(1),
        )
        return torch.cat([rgb, depth.unsqueeze(1), mask.unsqueeze(1)], dim=1)
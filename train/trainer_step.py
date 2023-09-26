from utils.loss import sem_loss_function, class_loss_function, pixel_anchoring_function
import torch
import torch.nn as nn
import os
import cv2
import numpy as np
import time


class TrainStepper():
    def __init__(self, deco_model, learning_rate, loss_weight, pal_loss_weight, device):
        self.device = device

        self.model = deco_model

        self.optimizer_sem = torch.optim.Adam(params=list(self.model.encoder_sem.parameters()) + list(self.model.decoder_sem.parameters()),
                                              lr=learning_rate, weight_decay=0.0001)
        self.optimizer_part = torch.optim.Adam(
            params=list(self.model.encoder_part.parameters()) + list(self.model.decoder_part.parameters()), lr=learning_rate,
            weight_decay=0.0001)
        self.optimizer_contact = torch.optim.Adam(
            params=list(self.model.encoder_sem.parameters()) + list(self.model.encoder_part.parameters()) + list(
                self.model.cross_att.parameters()) + list(self.model.classif.parameters()), lr=learning_rate, weight_decay=0.0001)

        self.sem_loss = sem_loss_function().to(device)
        self.class_loss = class_loss_function().to(device)
        self.pixel_anchoring_loss_smplx = pixel_anchoring_function(model_type='smplx').to(device)
        self.pixel_anchoring_loss_smpl = pixel_anchoring_function(model_type='smpl').to(device)
        self.lr = learning_rate
        self.loss_weight = loss_weight
        self.pal_loss_weight = pal_loss_weight

    def optimize(self, batch):
        self.model.train()

        img_paths = batch['img_path']
        img = batch['img'].to(self.device)

        img_scale_factor = batch['img_scale_factor'].to(self.device)

        pose = batch['pose'].to(self.device)
        betas = batch['betas'].to(self.device)
        transl = batch['transl'].to(self.device)
        has_smpl = batch['has_smpl'].to(self.device)
        is_smplx = batch['is_smplx'].to(self.device)

        cam_k = batch['cam_k'].to(self.device)

        gt_contact_labels_3d = batch['contact_label_3d'].to(self.device)
        has_contact_3d = batch['has_contact_3d'].to(self.device)

        sem_mask_gt = batch['sem_mask'].to(self.device)
        part_mask_gt = batch['part_mask'].to(self.device)

        polygon_contact_2d = batch['polygon_contact_2d'].to(self.device)
        has_polygon_contact_2d = batch['has_polygon_contact_2d'].to(self.device)

        # Forward pass
        cont, sem_mask_pred, part_mask_pred = self.model(img)

        loss_sem = self.sem_loss(sem_mask_gt, sem_mask_pred)
        loss_part = self.sem_loss(part_mask_gt, part_mask_pred)
        valid_contact_3d = has_contact_3d
        loss_cont = self.class_loss(gt_contact_labels_3d, cont, valid_contact_3d)
        valid_polygon_contact_2d = has_polygon_contact_2d

        if self.pal_loss_weight > 0 and (is_smplx == 0).sum() > 0:
            smpl_body_params = {'pose': pose[is_smplx == 0], 'betas': betas[is_smplx == 0],
                                'transl': transl[is_smplx == 0],
                                'has_smpl': has_smpl[is_smplx == 0]}
            loss_pix_anchoring_smpl, contact_2d_pred_rgb_smpl, _ = self.pixel_anchoring_loss_smpl(cont[is_smplx == 0],
                                                                                                  smpl_body_params,
                                                                                                  cam_k[is_smplx == 0],
                                                                                                  img_scale_factor[
                                                                                                      is_smplx == 0],
                                                                                                  polygon_contact_2d[
                                                                                                      is_smplx == 0],
                                                                                                  valid_polygon_contact_2d[
                                                                                                      is_smplx == 0])
            # weigh the smpl loss based on the number of smpl sample
            loss_pix_anchoring = loss_pix_anchoring_smpl * (is_smplx == 0).sum() / len(is_smplx)
            contact_2d_pred_rgb = contact_2d_pred_rgb_smpl
        else:
            loss_pix_anchoring = 0
            contact_2d_pred_rgb = torch.zeros_like(polygon_contact_2d)

        loss = loss_sem + loss_part + self.loss_weight * loss_cont + self.pal_loss_weight * loss_pix_anchoring

        self.optimizer_sem.zero_grad()
        self.optimizer_part.zero_grad()
        self.optimizer_contact.zero_grad()

        loss.backward()

        self.optimizer_sem.step()
        self.optimizer_part.step()
        self.optimizer_contact.step()

        losses = {'sem_loss': loss_sem,
                  'part_loss': loss_part,
                  'cont_loss': loss_cont,
                  'pal_loss': loss_pix_anchoring,
                  'total_loss': loss}

        output = {
            'img': img,
            'sem_mask_gt': sem_mask_gt,
            'sem_mask_pred': sem_mask_pred,
            'part_mask_gt': part_mask_gt,
            'part_mask_pred': part_mask_pred,
            'has_contact_2d': has_polygon_contact_2d,
            'contact_2d_gt': polygon_contact_2d,
            'contact_2d_pred_rgb': contact_2d_pred_rgb,
            'has_contact_3d': has_contact_3d,
            'contact_labels_3d_gt': gt_contact_labels_3d,
            'contact_labels_3d_pred': cont}

        return losses, output

    @torch.no_grad()
    def evaluate(self, batch):
        self.model.eval()

        img_paths = batch['img_path']
        img = batch['img'].to(self.device)

        img_scale_factor = batch['img_scale_factor'].to(self.device)

        pose = batch['pose'].to(self.device)
        betas = batch['betas'].to(self.device)
        transl = batch['transl'].to(self.device)
        has_smpl = batch['has_smpl'].to(self.device)
        is_smplx = batch['is_smplx'].to(self.device)

        cam_k = batch['cam_k'].to(self.device)

        gt_contact_labels_3d = batch['contact_label_3d'].to(self.device)
        has_contact_3d = batch['has_contact_3d'].to(self.device)

        sem_mask_gt = batch['sem_mask'].to(self.device)
        part_mask_gt = batch['part_mask'].to(self.device)

        polygon_contact_2d = batch['polygon_contact_2d'].to(self.device)
        has_polygon_contact_2d = batch['has_polygon_contact_2d'].to(self.device)

        # Forward pass
        initial_time = time.time()
        cont, sem_mask_pred, part_mask_pred = self.model(img)
        time_taken = time.time() - initial_time

        # time the following losses
        loss_sem = self.sem_loss(sem_mask_gt, sem_mask_pred)
        loss_part = self.sem_loss(part_mask_gt, part_mask_pred)
        valid_contact_3d = has_contact_3d
        loss_cont = self.class_loss(gt_contact_labels_3d, cont, valid_contact_3d)
        valid_polygon_contact_2d = has_polygon_contact_2d

        if self.pal_loss_weight > 0 and (is_smplx == 0).sum() > 0: # PAL loss only on 2D contacts in HOT which only has SMPL
            smpl_body_params = {'pose': pose[is_smplx == 0], 'betas': betas[is_smplx == 0], 'transl': transl[is_smplx == 0],
                                'has_smpl': has_smpl[is_smplx == 0]}
            loss_pix_anchoring_smpl, contact_2d_pred_rgb_smpl, _ = self.pixel_anchoring_loss_smpl(cont[is_smplx == 0],
                                                                                                 smpl_body_params,
                                                                                                 cam_k[is_smplx == 0],
                                                                                                 img_scale_factor[
                                                                                                     is_smplx == 0],
                                                                                                 polygon_contact_2d[
                                                                                                     is_smplx == 0],
                                                                                                 valid_polygon_contact_2d[
                                                                                                     is_smplx == 0])
            # weight the smpl loss based on the number of smpl samples
            contact_2d_pred_rgb = contact_2d_pred_rgb_smpl
            loss_pix_anchoring = loss_pix_anchoring_smpl * (is_smplx == 0).sum() / len(is_smplx)
        else:
            loss_pix_anchoring = 0
            contact_2d_pred_rgb = torch.zeros_like(polygon_contact_2d)

        loss = loss_sem + loss_part + self.loss_weight * loss_cont + self.pal_loss_weight * loss_pix_anchoring

        losses = {'sem_loss': loss_sem,
                  'part_loss': loss_part,
                  'cont_loss': loss_cont,
                  'pal_loss': loss_pix_anchoring,
                  'total_loss': loss}

        output = {
            'img': img,
            'sem_mask_gt': sem_mask_gt,
            'sem_mask_pred': sem_mask_pred,
            'part_mask_gt': part_mask_gt,
            'part_mask_pred': part_mask_pred,
            'has_contact_2d': has_polygon_contact_2d,
            'contact_2d_gt': polygon_contact_2d,
            'contact_2d_pred_rgb': contact_2d_pred_rgb,
            'has_contact_3d': has_contact_3d,
            'contact_labels_3d_gt': gt_contact_labels_3d,
            'contact_labels_3d_pred': cont}
        return losses, output, time_taken

    def save(self, ep, f1, model_path):
        # create model directory if it does not exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save({
            'epoch': ep,
            'deco': self.model.state_dict(),
            'f1': f1,
            'sem_optim': self.optimizer_sem.state_dict(),
            'part_optim': self.optimizer_part.state_dict(),
            'contact_optim': self.optimizer_contact.state_dict()
        },
            model_path)

    def load(self, model_path):
        print(f'~~~ Loading existing checkpoint from {model_path} ~~~')
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['deco'], strict=True)

        self.optimizer_sem.load_state_dict(checkpoint['sem_optim'])
        self.optimizer_part.load_state_dict(checkpoint['part_optim'])
        self.optimizer_contact.load_state_dict(checkpoint['contact_optim'])
        epoch = checkpoint['epoch']
        f1 = checkpoint['f1']
        return epoch, f1

    def update_lr(self, factor=2):
        if factor:
            new_lr = self.lr / factor

        self.optimizer_sem = torch.optim.Adam(params=list(self.model.encoder_sem.parameters()) + list(self.model.decoder_sem.parameters()),
                                              lr=new_lr, weight_decay=0.0001)
        self.optimizer_part = torch.optim.Adam(
            params=list(self.model.encoder_part.parameters()) + list(self.model.decoder_part.parameters()), lr=new_lr, weight_decay=0.0001)
        self.optimizer_contact = torch.optim.Adam(
            params=list(self.model.encoder_sem.parameters()) + list(self.model.encoder_part.parameters()) + list(
                self.model.cross_att.parameters()) + list(self.model.classif.parameters()), lr=new_lr, weight_decay=0.0001)

        print('update learning rate: %f -> %f' % (self.lr, new_lr))
        self.lr = new_lr
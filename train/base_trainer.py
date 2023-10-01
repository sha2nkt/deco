from tqdm import tqdm
from utils.metrics import metric, precision_recall_f1score, det_error_metric
import wandb
import torch
import numpy as np
from vis.visualize import gen_render


def trainer(epoch, train_loader, solver, hparams, compute_metrics=False):

    total_epochs = hparams.TRAINING.NUM_EPOCHS
    print('Training Epoch {}/{}'.format(epoch, total_epochs))

    length = len(train_loader)
    iterator = tqdm(enumerate(train_loader), total=length, leave=False, desc=f'Training Epoch: {epoch}/{total_epochs}')
    for step, batch in iterator:
        losses, output = solver.optimize(batch)
        # wandb logging every summary_steps steps
        if step % hparams.TRAINING.SUMMARY_STEPS == 0:
            iteration_summaries(batch, output, losses, mode='train')
    return losses, output

@torch.no_grad()
def evaluator(val_loader, solver, hparams, epoch=0, dataset_name='Unknown', normalize=True, return_dict=False, log_wandb=True):
    total_epochs = hparams.TRAINING.NUM_EPOCHS

    batch_size = val_loader.batch_size
    dataset_size = len(val_loader.dataset)
    print(f'Dataset size: {dataset_size}')

    val_epoch_cont_pre = np.zeros(dataset_size)
    val_epoch_cont_rec = np.zeros(dataset_size)
    val_epoch_cont_f1 = np.zeros(dataset_size)
    val_epoch_fp_geo_err = np.zeros(dataset_size)
    val_epoch_fn_geo_err = np.zeros(dataset_size)
    if hparams.TRAINING.CONTEXT:
        val_epoch_sem_iou = np.zeros(dataset_size)
        val_epoch_part_iou = np.zeros(dataset_size)

    val_epoch_cont_loss = np.zeros(dataset_size)
    
    total_time = 0

    rend_images = []

    eval_dict = {}

    length = len(val_loader)
    iterator = tqdm(enumerate(val_loader), total=length, leave=False, desc=f'Evaluating {dataset_name.capitalize()} Epoch: {epoch}/{total_epochs}')
    for step, batch in iterator:
        curr_batch_size = batch['img'].shape[0]
        losses, output, time_taken = solver.evaluate(batch)

        val_epoch_cont_loss[step * batch_size:step * batch_size + curr_batch_size] = losses['cont_loss'].cpu().numpy()

        # compute metrics
        contact_labels_3d = output['contact_labels_3d_gt']
        has_contact_3d = output['has_contact_3d']
        # check if any value in has_contact_3d tensor is 0
        assert torch.any(has_contact_3d == 0) == False, 'has_contact_3d tensor has 0 values'

        contact_labels_3d_pred = output['contact_labels_3d_pred']
        if hparams.TRAINING.CONTEXT:
            sem_mask_gt = output['sem_mask_gt']
            sem_seg_pred = output['sem_mask_pred']
            part_mask_gt = output['part_mask_gt']
            part_seg_pred = output['part_mask_pred']

        cont_pre, cont_rec, cont_f1 = precision_recall_f1score(contact_labels_3d, contact_labels_3d_pred)
        fp_geo_err, fn_geo_err = det_error_metric(contact_labels_3d_pred, contact_labels_3d)
        if hparams.TRAINING.CONTEXT:
            sem_iou = metric(sem_mask_gt, sem_seg_pred)
            part_iou = metric(part_mask_gt, part_seg_pred)

        val_epoch_cont_pre[step * batch_size:step * batch_size + curr_batch_size] = cont_pre.cpu().numpy()
        val_epoch_cont_rec[step * batch_size:step * batch_size + curr_batch_size] = cont_rec.cpu().numpy()
        val_epoch_cont_f1[step * batch_size:step * batch_size + curr_batch_size] = cont_f1.cpu().numpy()
        val_epoch_fp_geo_err[step * batch_size:step * batch_size + curr_batch_size] = fp_geo_err.cpu().numpy()
        val_epoch_fn_geo_err[step * batch_size:step * batch_size + curr_batch_size] = fn_geo_err.cpu().numpy()
        if hparams.TRAINING.CONTEXT:
            val_epoch_sem_iou[step * batch_size:step * batch_size + curr_batch_size] = sem_iou.cpu().numpy()
            val_epoch_part_iou[step * batch_size:step * batch_size + curr_batch_size] = part_iou.cpu().numpy()
        
        total_time += time_taken

        # wandb logging every summary_steps steps
        if step % hparams.VALIDATION.SUMMARY_STEPS == 0:
            if hparams.TRAINING.CONTEXT:
                rend = gen_render(output, normalize)
                rend_images.append(rend)
            if log_wandb: 
                iteration_summaries(batch, output, losses, dataset_name, mode='val')

    eval_dict['cont_precision'] = np.sum(val_epoch_cont_pre) / dataset_size
    eval_dict['cont_recall'] = np.sum(val_epoch_cont_rec) / dataset_size
    eval_dict['cont_f1'] = np.sum(val_epoch_cont_f1) / dataset_size
    eval_dict['fp_geo_err'] = np.sum(val_epoch_fp_geo_err) / dataset_size
    eval_dict['fn_geo_err'] = np.sum(val_epoch_fn_geo_err) / dataset_size
    if hparams.TRAINING.CONTEXT:
        eval_dict['sem_iou'] = np.sum(val_epoch_sem_iou) / dataset_size
        eval_dict['part_iou'] = np.sum(val_epoch_part_iou) / dataset_size
        eval_dict['images'] = rend_images
    
    total_time /= dataset_size
    
    if log_wandb: 
        epoch_summaries(eval_dict, hparams.TRAINING.CONTEXT, dataset_name, mode='val', epoch=epoch)
        epoch_summaries_best(eval_dict, dataset_name, mode='val')

    val_epoch_cont_loss = np.sum(val_epoch_cont_loss) / dataset_size
    if return_dict:
        return eval_dict, total_time
    return eval_dict['cont_f1']

def iteration_summaries(input_batch, output, losses, dataset_name='train-data', mode='train'):
    for loss_name, val in losses.items():
        wandb.log({f'{mode}_loss/{dataset_name}__' + loss_name: val})

def epoch_summaries(output, context, dataset_name, mode='val', epoch=0):
    for out_key, val in output.items():
        if out_key != f'images':
            wandb.log({f'{mode}/{dataset_name}__' + out_key: val,
                       f'epoch': epoch})
    if context:
        wandb.log({f'{mode}/{dataset_name}__' + 'images': [wandb.Image(img) for img in output['images']],
                f'epoch': epoch})

def epoch_summaries_best(output, dataset_name, mode='val'):
    for out_key, val in output.items():
        try:
            if out_key != f'images':
                if 'geo_err' in out_key:
                    if wandb.run.summary[f'{mode}Best/{dataset_name}__' + out_key] > wandb.run.summary[f'{mode}/{dataset_name}__' + out_key]:
                        wandb.run.summary[f'{mode}Best/{dataset_name}__' + out_key] = wandb.run.summary[f'{mode}/{dataset_name}__' + out_key]
                else:
                    if wandb.run.summary[f'{mode}Best/{dataset_name}__' + out_key] < wandb.run.summary[f'{mode}/{dataset_name}__' + out_key]:
                        wandb.run.summary[f'{mode}Best/{dataset_name}__' + out_key] = wandb.run.summary[f'{mode}/{dataset_name}__' + out_key]
        except KeyError:
            if out_key != f'images':
                wandb.run.summary[f'{mode}Best/{dataset_name}__' + out_key] = wandb.run.summary[f'{mode}/{dataset_name}__' + out_key]
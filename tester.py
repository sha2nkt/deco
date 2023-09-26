import torch
from torch.utils.data import DataLoader
from loguru import logger

from train.trainer_step import TrainStepper
from train.base_trainer import evaluator
from data.base_dataset import BaseDataset
from models.deco import DECO
from utils.config import parse_args, run_grid_search_experiments

def test(hparams):
    deco_model = DECO(hparams.TRAINING.ENCODER, device)
    pytorch_total_params = sum(p.numel() for p in deco_model.parameters() if p.requires_grad)
    print('Total number of trainable parameters: ', pytorch_total_params)

    solver = TrainStepper(deco_model, hparams.OPTIMIZER.LR, hparams.TRAINING.LOSS_WEIGHTS, hparams.TRAINING.PAL_LOSS_WEIGHTS, device)

    logger.info(f'Loading weights from {hparams.TRAINING.BEST_MODEL_PATH}')
    _, _ = solver.load(hparams.TRAINING.BEST_MODEL_PATH)
    
    # Run testing
    for test_loader in val_loaders:
        dataset_name = test_loader.dataset.dataset
        test_dict, total_time = evaluator(test_loader, solver, hparams, 0, dataset_name, return_dict=True, log_wandb=False)

        print('Test Contact Precision: ', test_dict['cont_precision'])
        print('Test Contact Recall: ', test_dict['cont_recall'])
        print('Test Contact F1 Score: ', test_dict['cont_f1'])
        print('Test Contact FP Geo. Error: ', test_dict['fp_geo_err'])
        print('Test Contact FN Geo. Error: ', test_dict['fn_geo_err'])
        print('Test Contact Semantic Segmentation IoU: ', test_dict['sem_iou'])
        print('Test Contact Part Segmentation IoU: ', test_dict['part_iou'])
        print('\nTime taken per image for evaluation: ', total_time)
        print('-'*50)

if __name__ == '__main__':
    args = parse_args()
    hparams = run_grid_search_experiments(
        args,
        script='tester.py',
        change_wt_name=False
    )

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    val_datasets = []
    for ds in hparams.VALIDATION.DATASETS:
        if ds in ['rich', 'prox']:
            val_datasets.append(BaseDataset(ds, 'val', model_type='smplx', normalize=hparams.DATASET.NORMALIZE_IMAGES))
        elif ds in ['hot', 'hot_nosupport', 'dca', 'hot_behave', 'hot_phosa']:
            val_datasets.append(BaseDataset(ds, 'val', model_type='smpl', normalize=hparams.DATASET.NORMALIZE_IMAGES))
        else:
            raise ValueError('Dataset not supported')

    val_loaders = [DataLoader(val_dataset, batch_size=hparams.DATASET.BATCH_SIZE, shuffle=False, num_workers=hparams.DATASET.NUM_WORKERS) for val_dataset in val_datasets]

    test(hparams)
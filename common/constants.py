from os.path import join

smpl_dir = '/is/cluster/fast/achatterjee/dca_contact/'
DIST_MATRIX_PATH = 'data/smpl/smpl_neutral_geodesic_dist.npy'
SMPL_MEAN_PARAMS = join(smpl_dir, 'data/smpl_mean_params.npz')
SMPL_MODEL_DIR = join(smpl_dir, 'data/smpl/')
SMPLX_MODEL_DIR = join(smpl_dir, 'data/smplx/')

N_PARTS = 24

# Mean and standard deviation for normalizing input image
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]

# Output folder to save test/train npz files
DATASET_NPZ_PATH = './data/Datasets'
CONTACT_MAPPING_PATH = '/ps/scratch/ps_shared/stripathi/deco/4agniv/essentials/models_utils/'

# Path to test/train npz files
DATASET_FILES = {
    'train': {
        'hot': join(DATASET_NPZ_PATH, 'hot/npzs/hot_noprox_trainval_combined.npz'),
        'hot_nosupport': join(DATASET_NPZ_PATH, 'hot/npzs/hot_noprox_supporting_False_trainval_combined.npz'),
        'rich': join(DATASET_NPZ_PATH, 'rich/final_npzs/rich_train_smplx_cropped_bmp.npz'),
        'prox': join(DATASET_NPZ_PATH, 'prox/npzs/prox_train_smplx_ds4.npz'),
        'dca': join(DATASET_NPZ_PATH, 'dca/npzs/hot_dca_trainval.npz'),
    },
    'val': {
        'hot': join(DATASET_NPZ_PATH, 'hot/npzs/hot_dca_test.npz'),
        'hot_nosupport': join(DATASET_NPZ_PATH, 'hot/npzs/hot_dca_supporting_False_test.npz'),
        'hot_behave': join(DATASET_NPZ_PATH, 'hot_behave/hot_behave_test.npz'),
        'hot_phosa': join(DATASET_NPZ_PATH, 'hot_phosa/hot_phosa_test.npz'),
        'rich': join(DATASET_NPZ_PATH, 'rich/final_npzs/rich_test_smplx_cropped_bmp.npz'),
        'prox': join(DATASET_NPZ_PATH, 'prox/npzs/prox_val_smplx_ds4.npz'),
    },
    'test': {
        'hot': join(DATASET_NPZ_PATH, 'hot/npzs/hot_dca_test.npz'),
        'hot_behave': join(DATASET_NPZ_PATH, 'hot_behave/hot_behave_test.npz'),
        'rich': join(DATASET_NPZ_PATH, 'rich/final_npzs/rich_test_smplx_cropped_bmp.npz'),
        'prox': join(DATASET_NPZ_PATH, 'prox/npzs/prox_val_smplx_ds4.npz'),
    },
}
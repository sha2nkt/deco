from os.path import join

DIST_MATRIX_PATH = 'data/smpl/smpl_neutral_geodesic_dist.npy'
SMPL_MEAN_PARAMS = 'data/smpl_mean_params.npz'
SMPL_MODEL_DIR = 'data/smpl/'
SMPLX_MODEL_DIR = 'data/smplx/'

N_PARTS = 24

# Mean and standard deviation for normalizing input image
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]

# Output folder to save test/train npz files
DATASET_NPZ_PATH = 'datasets/Release_Datasets'
CONTACT_MAPPING_PATH = 'data/conversions'

# Path to test/train npz files
DATASET_FILES = {
    'train': {
        'damon': join(DATASET_NPZ_PATH, 'damon/hot_dca_trainval.npz'),
        'rich': join(DATASET_NPZ_PATH, 'rich/rich_train_smplx_cropped_bmp.npz'),
        'prox': join(DATASET_NPZ_PATH, 'prox/prox_train_smplx_ds4.npz'),
    },
    'val': {
        'damon': join(DATASET_NPZ_PATH, 'damon/hot_dca_test.npz'),
        'rich': join(DATASET_NPZ_PATH, 'rich/rich_test_smplx_cropped_bmp.npz'),
        'prox': join(DATASET_NPZ_PATH, 'prox/prox_val_smplx_ds4.npz'),
    },
    'test': {
        'damon': join(DATASET_NPZ_PATH, 'damon/hot_dca_test.npz'),
        'rich': join(DATASET_NPZ_PATH, 'rich/rich_test_smplx_cropped_bmp.npz'),
        'prox': join(DATASET_NPZ_PATH, 'prox/prox_val_smplx_ds4.npz'),
    },
}

categorization_unordered = {
    "transportation": [
        'motorcycle', 'bicycle', 'boat', 'car', 'truck', 'bus', 'train', 'airplane',],
    "accessories": [
        'backpack', 'tie', 'handbag', 'baseball glove'],
    "furniture": [
        'bench', 'chair', 'couch', 'bed', 'toilet', 'dining table'],
    'everyday objects': [
        'book', 'umbrella', 'cell phone', 'laptop', 'kite', 'suitcase', 'bottle', 'remote',
        'toothbrush', 'teddy bear', 'scissors', 'keyboard', 'hair drier', 'traffic light',
        'fire hydrant', 'stop sign', 'tv', 'vase', 'parking meter', 'clock', 'potted plant',
        'mouse'],
    'sports equipment': [
        'frisbee', 'sports ball', 'tennis racket', 'baseball bat',
        'skateboard', 'snowboard', 'skis', 'surfboard'],
    'food items': [
        'banana', 'cake', 'apple', 'carrot', 'pizza', 'donut', 'hot dog',
        'sandwich', 'broccoli', 'orange'],
    'kitchen appliances': [
        'knife', 'spoon', 'cup', 'wine glass', 'oven', 'fork', 'bowl',
        'refrigerator', 'toaster', 'sink', 'microwave']}

object_names = ['motorcycle', 'bicycle', 'boat', 'car', 'truck', 'bus', 'train', 'airplane', 'backpack', 'tie',
                'handbag', 'baseball glove', 'bench', 'chair', 'couch', 'bed', 'toilet', 'dining table', 'book',
                'umbrella', 'cell phone', 'laptop', 'kite', 'suitcase', 'bottle', 'remote', 'toothbrush', 'teddy bear',
                'scissors', 'keyboard', 'hair drier', 'traffic light', 'fire hydrant', 'stop sign', 'tv', 'vase',
                'parking meter', 'clock', 'potted plant', 'mouse', 'frisbee', 'sports ball', 'tennis racket',
                'baseball bat', 'skateboard', 'snowboard', 'skis', 'surfboard', 'banana', 'cake', 'apple', 'carrot',
                'pizza', 'donut', 'hot dog', 'sandwich', 'broccoli', 'orange', 'knife', 'spoon', 'cup', 'wine glass',
                'oven', 'fork', 'bowl', 'refrigerator', 'toaster', 'sink', 'microwave']

# replace spaces in object names with underscores
object_names = [name.replace(' ', '_') for name in object_names]

# replaces spaces in categorization_unordered values with underscores
categorization_unordered = {key: [name.replace(' ', '_') for name in value] for key, value in categorization_unordered.items()}

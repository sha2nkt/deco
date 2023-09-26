import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import cv2
import pandas as pd
import numpy as np

# from data.loader import BaseDataset
from models.components import Encoder, Cross_Att, Classifier
from utils.metrics import precision_recall_f1score
from vis.vis_comp import gen_render

dataset_root_path = '/is/cluster/work/achatterjee/rich/vis'
model_path = '/is/cluster/work/achatterjee/weights/hypparam_tune/lw/lw2_best.pt'
names = sorted(os.listdir('/is/cluster/work/achatterjee/rich/vis/images'))


def mask_split(img, num_parts):
    img = img[:, :, 0]
    mask = np.zeros((img.shape[0], img.shape[1], num_parts))
    for i in np.unique(img):
        mask[:, :, i] = np.where(img == i, 1., 0.)
    return np.transpose(mask, (2, 0, 1))


def loader(img_path, mask_path, sem_mask_path, part_mask_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256), cv2.INTER_CUBIC)
    img = img.transpose(2, 0, 1) / 255.0

    df = pd.read_pickle(mask_path)
    mask = df['contact']

    sem_mask = cv2.imread(sem_mask_path)
    sem_mask = cv2.resize(sem_mask, (256, 256), cv2.INTER_CUBIC)
    sem_mask = mask_split(sem_mask, 133)

    part_mask = cv2.imread(part_mask_path)
    part_mask = cv2.resize(part_mask, (256, 256), cv2.INTER_CUBIC)
    part_mask = mask_split(part_mask, 20)

    img = torch.tensor(img, dtype=torch.float32)
    mask = torch.unsqueeze(torch.tensor(mask, dtype=torch.float32), 0)
    sem_mask = torch.tensor(sem_mask, dtype=torch.float32)
    part_mask = torch.tensor(part_mask, dtype=torch.float32)

    return img, mask, sem_mask, part_mask


def read_dataset(root_path):
    images = []
    masks = []
    sem_masks = []
    part_masks = []

    image_root = os.path.join(root_path, 'images')
    gt_root = os.path.join(root_path, 'labels')
    sem_root = os.path.join(root_path, 'segmentation_masks')
    part_root = os.path.join(root_path, 'parts')

    for image_name in sorted(os.listdir(image_root)):
        image_path = os.path.join(image_root, image_name)
        images.append(image_path)
    for mask_name in sorted(os.listdir(gt_root)):
        mask_path = os.path.join(gt_root, mask_name)
        masks.append(mask_path)
    for sem_mask_name in sorted(os.listdir(sem_root)):
        sem_mask_path = os.path.join(sem_root, sem_mask_name)
        sem_masks.append(sem_mask_path)
    for part_mask_name in sorted(os.listdir(part_root)):
        part_mask_path = os.path.join(part_root, part_mask_name)
        part_masks.append(part_mask_path)

    return images, masks, sem_masks, part_masks


class Dataset(Dataset):

    def __init__(self, root_path):
        self.root = root_path
        self.images, self.labels, self.sem_masks, self.part_masks = read_dataset(self.root)

    def __getitem__(self, index):
        img, mask, sem_mask, part_mask = loader(self.images[index], self.labels[index], self.sem_masks[index],
                                                self.part_masks[index])
        return img, mask, sem_mask, part_mask

    def __len__(self):
        return len(self.images)


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

test_dataset = Dataset(dataset_root_path)

print(len(test_dataset))
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

file_names = sorted(os.listdir(os.path.join(dataset_root_path, 'images')))


class MyFrame():
    def __init__(self, enc_sem, enc_part, ca, classifier, device):
        self.enc_sem = enc_sem.to(device)
        self.enc_part = enc_part.to(device)
        self.ca = ca.to(device)
        self.classifier = classifier.to(device)

    def set_input(self, img_batch, label_batch):
        self.img = img_batch
        self.label = label_batch

    def optimize(self):
        sem_enc_out = self.enc_sem.forward(self.img)
        part_enc_out = self.enc_part.forward(self.img)

        sem_enc_out = torch.reshape(sem_enc_out, (-1, 1, 1024))
        part_enc_out = torch.reshape(part_enc_out, (-1, 1, 1024))

        cont = self.classifier.forward(self.ca.forward(sem_enc_out, part_enc_out))

        torch.backends.cudnn.benchmark = True

        return cont

    def load(self, model_path):
        checkpoint = torch.load(model_path)
        self.enc_sem.load_state_dict(checkpoint['sem_enc'])
        self.enc_part.load_state_dict(checkpoint['part_enc'])
        self.ca.load_state_dict(checkpoint['ca'])
        self.classifier.load_state_dict(checkpoint['classifier'])


@torch.no_grad()
def tester(test_loader, solver):
    test_epoch_cont_pre = 0
    test_epoch_cont_rec = 0
    test_epoch_cont_f1 = 0

    f1_list = []

    length = len(test_loader)
    iterator = tqdm(enumerate(test_loader), total=length, leave=False, desc='Testing...')
    for idx, (img, mask, _, _) in iterator:
        img = img.to(device)
        mask = mask.to(device)
        solver.set_input(img, mask)
        cont = solver.optimize()

        img = img.detach().cpu()
        mask = mask.detach().cpu()
        cont = cont.detach().cpu()

        gen_render(img.numpy(), cont.numpy(), names[idx], mask.numpy())

        cont_pre, cont_rec, cont_f1 = precision_recall_f1score(mask, cont)

        test_epoch_cont_pre += cont_pre
        test_epoch_cont_rec += cont_rec
        test_epoch_cont_f1 += cont_f1

        if (test_epoch_cont_pre + test_epoch_cont_rec) == 0:
            f1 = 2 * test_epoch_cont_pre * test_epoch_cont_rec / (test_epoch_cont_pre + test_epoch_cont_rec + 1e-20)
        else:
            f1 = 2 * test_epoch_cont_pre * test_epoch_cont_rec / (test_epoch_cont_pre + test_epoch_cont_rec)

        f1_list.append(f1)

    test_epoch_cont_pre = test_epoch_cont_pre / len(test_dataset)
    test_epoch_cont_rec = test_epoch_cont_rec / len(test_dataset)
    test_epoch_cont_f1 = test_epoch_cont_f1 / len(test_dataset)

    return test_epoch_cont_pre, test_epoch_cont_rec, test_epoch_cont_f1, f1_list


def test():
    encoder_sem = Encoder()
    encoder_part = Encoder()
    cross_att = Cross_Att(1024, 1024)
    classif = Classifier(1024)

    solver = MyFrame(encoder_sem, encoder_part, cross_att, classif, device)
    solver.load(model_path)

    tcp, tcr, tcf1, f1_list = tester(test_loader, solver)

    N = 20
    names = []

    res = sorted(range(len(f1_list)), key=lambda sub: f1_list[sub])[-N:]

    for i in res:
        names.append(file_names[i])

    print('Test Contact Precision: ', tcp)
    print('Test Contact Recall: ', tcr)
    print('Test Contact F1 Score: ', tcf1)

    print('Names: ', names)

    print('----------------------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------------------')


test()

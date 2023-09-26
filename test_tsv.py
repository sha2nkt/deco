import torch
from tqdm import tqdm
import cv2
import os
import json
import numpy as np

from models.components import Encoder, Cross_Att, Classifier
from utils.loss import class_loss_function
from utils.metrics import precision_recall_f1score

model_path = '/is/cluster/work/achatterjee/weights/hypparam_tune/lw/lw2_best.pt'
root = '/ps/project/datasets/RICH/website_release/images/test'

file_path = '/is/cluster/work/achatterjee/split/test.json'
file_name = open(file_path, 'r')
test_json = json.load(file_name)
file_name.close()
img_names = test_json['Images']
l = len(img_names)
print(l)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def loader(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256), cv2.INTER_CUBIC)
    img = img.transpose(2, 0, 1) / 255.0
    img = img[np.newaxis, :, :, :]
    img = torch.tensor(img, dtype=torch.float32)

    return img


class MyFrame():
    def __init__(self, enc_sem, enc_part, ca, classifier, device):
        self.enc_sem = enc_sem.to(device)
        self.enc_part = enc_part.to(device)
        self.ca = ca.to(device)
        self.classifier = classifier.to(device)

        self.class_loss = class_loss_function().to(device)

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

        loss_cont = self.class_loss(self.label, cont)

        return loss_cont, cont

    def load(self, model_path):
        checkpoint = torch.load(model_path)
        self.enc_sem.load_state_dict(checkpoint['sem_enc'])
        self.enc_part.load_state_dict(checkpoint['part_enc'])
        self.ca.load_state_dict(checkpoint['ca'])
        self.classifier.load_state_dict(checkpoint['classifier'])


@torch.no_grad()
def tester(solver):
    test_epoch_cont_loss = 0

    test_epoch_cont_pre = 0
    test_epoch_cont_rec = 0
    test_epoch_cont_f1 = 0

    for idx, img_name in tqdm(enumerate(img_names), dynamic_ncols=True):
        img_path = os.path.join(root, img_name)
        img = loader(img_path)
        img = img.to(device)

        gt_verts = test_json['Labels'][idx]
        mask = torch.zeros(1, 1, 6890)
        for i in gt_verts:
            mask[0, 0, i] = 1.
        mask = mask.to(device)

        solver.set_input(img, mask)
        cont_loss, cont = solver.optimize()

        img = img.detach().cpu()
        mask = mask.detach().cpu()
        cont = cont.detach().cpu()
        cont_loss = cont_loss.detach().cpu()

        test_epoch_cont_loss += cont_loss.numpy()

        cont_pre, cont_rec, cont_f1 = precision_recall_f1score(mask, cont)

        test_epoch_cont_pre += cont_pre
        test_epoch_cont_rec += cont_rec
        test_epoch_cont_f1 += cont_f1

    test_epoch_cont_loss = test_epoch_cont_loss / l

    test_epoch_cont_pre = test_epoch_cont_pre / l
    test_epoch_cont_rec = test_epoch_cont_rec / l
    test_epoch_cont_f1 = test_epoch_cont_f1 / l

    return test_epoch_cont_loss, test_epoch_cont_pre, test_epoch_cont_rec, test_epoch_cont_f1


def test():
    encoder_sem = Encoder()
    encoder_part = Encoder()
    cross_att = Cross_Att(1024, 1024)
    classif = Classifier(1024)

    solver = MyFrame(encoder_sem, encoder_part, cross_att, classif, device)
    solver.load(model_path)

    tcl, tcp, tcr, tcf1 = tester(solver)

    print('Test Contact Loss: ', tcl)

    print('Test Contact Precision: ', tcp)
    print('Test Contact Recall: ', tcr)
    print('Test Contact F1 Score: ', tcf1)

    print('----------------------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------------------')


test()

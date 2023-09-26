import torch
import os
import cv2
import numpy as np

from models.components import Encoder, Cross_Att, Classifier
from vis.vis_eval import gen_render

image_path = '/is/cluster/work/achatterjee/pose_estim/images/sit.png'
save_path = '/is/cluster/work/achatterjee/pose_estim/renders'
model_path = '/is/cluster/work/achatterjee/weights/hypparam_tune/lw/lw2_best.pt'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def loader(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256), cv2.INTER_CUBIC)
    img = img.transpose(2,0,1)/255.0
    img = img[np.newaxis, :, :, :]
    img = torch.tensor(img, dtype = torch.float32)
    
    return img
        
def predict(img, enc_sem, enc_part, ca, classifier):
    sem_enc_out = enc_sem.forward(img)
    part_enc_out = enc_part.forward(img)

    sem_enc_out = torch.reshape(sem_enc_out, (-1, 1, 1024))
    part_enc_out = torch.reshape(part_enc_out, (-1, 1, 1024))

    cont = classifier.forward(ca.forward(sem_enc_out, part_enc_out))

    torch.backends.cudnn.benchmark = True
      
    return cont

@torch.no_grad()
def inference():
    encoder_sem = Encoder().to(device)
    encoder_part = Encoder().to(device)
    cross_att = Cross_Att(1024, 1024).to(device)
    classif = Classifier(1024).to(device)

    checkpoint = torch.load(model_path)
    encoder_sem.load_state_dict(checkpoint['sem_enc'])
    encoder_part.load_state_dict(checkpoint['part_enc'])
    cross_att.load_state_dict(checkpoint['ca'])
    classif.load_state_dict(checkpoint['classifier'])

    img = loader(image_path)
    img = img.to(device)

    pred = predict(img, encoder_sem, encoder_part, cross_att, classif)

    rend = gen_render(img.detach().cpu().numpy(), pred.detach().cpu().numpy())
    save_name = os.path.basename(image_path)[:-4] + '_render.png'
    rend.save(os.path.join(save_path, save_name))

    return

inference()        
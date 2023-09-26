import torch
import torch.nn as nn
import numpy as np

from models.components import Encoder, Self_Attn, Decoder, Classifier

class DCA(nn.Module):
    def __init__(self):
        super(DCA, self).__init__()
        
        self.sa_sem_seg = Encoder(pretrained=False)
        self.sa_part_seg = Encoder(pretrained=False)

        self.sem_decoder = Decoder(1, 133)
        self.part_decoder = Decoder(1, 20)

        self.cross_attn_1 = Self_Attn(1024, 1024)
        self.cross_attn_2 = Self_Attn(1024, 1024)

        self.classifier = Classifier(1024, 6890)

    def forward(self, x):
        sem_seg = self.sa_sem_seg(x)
        part_seg = self.sa_part_seg(x)

        sem_seg_out = torch.reshape(sem_seg, (1, 1, 32, 32))
        part_seg_out = torch.reshape(part_seg, (1, 1, 32, 32))

        sem_seg_out = self.sem_decoder(sem_seg_out)
        part_seg_out = self.part_decoder(part_seg_out)

        sem_seg = torch.reshape(sem_seg, (1, 1, 1024))
        part_seg = torch.reshape(part_seg, (1, 1, 1024))

        cross1 = self.cross_attn_1(sem_seg, part_seg, part_seg)
        cross2 = self.cross_attn_1(part_seg, sem_seg, sem_seg)

        out = cross1 * cross2
        out = self.classifier(out)

        return sem_seg_out, part_seg_out, out
from models.components import Encoder, Cross_Att, Decoder, Classifier
import torch.nn as nn
import torch

class DECO(nn.Module):
    def __init__(self, encoder, context, device):
        super(DECO, self).__init__()
        self.encoder_type = encoder
        self.context = context

        self.encoder_sem = Encoder(encoder=encoder).to(device)
        self.encoder_part = Encoder(encoder=encoder).to(device)
        if self.encoder_type == 'hrnet':
            if self.context:    
                self.decoder_sem = Decoder(480, 133, encoder=encoder).to(device)
                self.decoder_part = Decoder(480, 26, encoder=encoder).to(device)
            self.sem_pool = nn.AdaptiveAvgPool2d((1))
            self.part_pool = nn.AdaptiveAvgPool2d((1))
            self.cross_att = Cross_Att(480, 480).to(device)
            self.classif = Classifier(480).to(device)
        elif self.encoder_type == 'swin':
            self.correction_conv = nn.Conv1d(768, 1024, 1).to(device)
            if self.context:    
                self.decoder_sem = Decoder(1, 133, encoder=encoder).to(device)
                self.decoder_part = Decoder(1, 26, encoder=encoder).to(device)
            self.cross_att = Cross_Att(1024, 1024).to(device)
            self.classif = Classifier(1024).to(device)
        else:
            NotImplementedError('Encoder type not implemented')

        self.device = device

    def forward(self, img):
        if self.encoder_type == 'hrnet':
            sem_enc_out = self.encoder_sem(img)
            part_enc_out = self.encoder_part(img)

            if self.context:
                sem_mask_pred = self.decoder_sem(sem_enc_out)
                part_mask_pred = self.decoder_part(part_enc_out)

            sem_enc_out = self.sem_pool(sem_enc_out)
            sem_enc_out = sem_enc_out.squeeze(2)
            sem_enc_out = sem_enc_out.squeeze(2)
            sem_enc_out = sem_enc_out.unsqueeze(1)

            part_enc_out = self.part_pool(part_enc_out)
            part_enc_out = part_enc_out.squeeze(2)
            part_enc_out = part_enc_out.squeeze(2)
            part_enc_out = part_enc_out.unsqueeze(1)

            att = self.cross_att(sem_enc_out, part_enc_out)
            cont = self.classif(att)
        else:
            sem_enc_out = self.encoder_sem(img)
            part_enc_out = self.encoder_part(img)

            sem_seg = torch.reshape(sem_enc_out, (-1, 768, 1))		
            part_seg = torch.reshape(part_enc_out, (-1, 768, 1))		

            sem_seg = self.correction_conv(sem_seg)		
            part_seg = self.correction_conv(part_seg)		

            sem_seg = torch.reshape(sem_seg, (-1, 1, 32, 32))		
            part_seg = torch.reshape(part_seg, (-1, 1, 32, 32))

            if self.context:
                sem_mask_pred = self.decoder_sem(sem_seg)
                part_mask_pred = self.decoder_part(part_seg)

            sem_enc_out = torch.reshape(sem_seg, (-1, 1, 1024))
            part_enc_out = torch.reshape(part_seg, (-1, 1, 1024))

            att = self.cross_att(sem_enc_out, part_enc_out)
            cont = self.classif(att)

        if self.context: return cont, sem_mask_pred, part_mask_pred
        return cont
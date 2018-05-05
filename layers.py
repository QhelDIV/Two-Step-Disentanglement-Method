import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import numpy as np

from config import VERBOSE

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class ConvLayer(nn.Module):
    def __init__(self, in_channel,  conv_channel, filter_size, stride=1, padding = 0, bn=False, upsampling=False):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channel, conv_channel, filter_size, stride=stride, padding = padding)
        # nn.init package contains convenient initialization methods
        # http://pytorch.org/docs/master/nn.html#torch-nn-init 
        nn.init.kaiming_normal_(self.conv.weight)
        
        self.use_bn = bn
        self.upsampling = upsampling
        if bn == True:
            self.bn = nn.BatchNorm2d(conv_channel)
        if upsampling==True:
            self.upsampling = True
            self.Upsampler = nn.Upsample(scale_factor=2, mode='bilinear')
        
    def forward(self, x):
        if VERBOSE:
            print(x.shape)
        x = self.conv(x)
        if self.upsampling==True:
            x = self.Upsampler(x)
        if self.use_bn == True:
            x = self.bn(x)
        x = F.relu(x)
        return x
    
class Dense(nn.Module):
    def __init__(self, in_channel,  out_channel, bn=False):
        super().__init__()
        
        self.dense = nn.Linear(in_channel, out_channel)
        nn.init.kaiming_normal_(self.dense.weight)
        
        self.use_bn = bn
        if bn == True:
            self.bn = nn.BatchNorm1d(out_channel)
    def forward(self, x):
        if VERBOSE:
            print(x.shape)
            
        x = self.dense(x)
        if self.use_bn == True:
            x = self.bn(x)
        x = F.relu(x)
        return x


def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        
        self.new_shape = shape
    def forward(self, x):
        return x.view(*self.new_shape)
def set_trainable(model, trainable = True):
    for param in model.parameters():
        param.requires_grad = trainable
class DisentNet(nn.Module):
    def __init__(self, Z_Encoder, SZ_Decoder):
        super().__init__()
        
        self.z_enc = Z_Encoder
        self.decoder=SZ_Decoder
        
        self.set_trainable(trainable=True)
    def get_z_encoder(self):
        return self.z_enc
    def get_decoder(self):
        return self.decoder
    def set_trainable(self, trainable=True):
        if trainable==True:
            set_trainable(self.z_enc,True)
            set_trainable(self.decoder,True)
        else:
            set_trainable(self.z_enc,False)
            set_trainable(self.decoder,False)
    def forward(self, s_latent, x):
        z_latent = self.z_enc(x).detach()
        latent = torch.cat((s_latent,z_latent),dim=1)
        reconstructed = self.decoder(latent)
        return reconstructed
        
class AdvNet(nn.Module):
    def __init__(self, Z_Encoder, Adv):
        super().__init__()
        
        self.z_enc = Z_Encoder
        self.adv = Adv
        
        self.set_trainable(trainable=True)
    def set_trainable(self, trainable=True):
        #set_trainable(self.z_enc,False)
        if trainable==True:
            set_trainable(self.adv,True)
        else:
            set_trainable(self.adv,False)
        
    def forward(self, x):
        z_latent = self.z_enc(x).detach()
        scores = self.adv(z_latent)
        return scores

        
################## TEST LAYERS ####################
class DecoderTest(nn.Module):
    # the goal of this class is to check whether decoder works or not
    # basically its an auto-encoder
    def __init__(self, S_Encoder, SZ_Decoder):
        super().__init__()
        self.encoder = S_Encoder
        self.decoder = SZ_Decoder
    def forward(self, x):
        s1_latent = self.encoder(x).detach()
        s2_latent = s1_latent
        
        latent = torch.cat((s1_latent,s1_latent),dim=1)
        
        reconstructed = self.decoder(latent)
        
        return reconstructed
        


# some setup code
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn import init

from config import *

import layers

from params import *


def Encoder(img_size, in_channel, conv_channel, filter_size, latent_dim, dense_size, bn):
    inner_conv_channel = conv_channel//2
    if img_size%4 != 0:
        print("WARNING: image size mod 4 != 0, may produce bug.")
    # total input number of the input of the last conv layer, new image size = old / 2 / 2 
    flatten_img_size = inner_conv_channel * img_size/4 * img_size/4
    
    # explain: first two layer's padding = 2, because we set W/S = W/S + floor((-F+2P)/S+1), S=2,F=5,so P=2
    if VERBOSE:
        print(img_size, in_channel, conv_channel, filter_size, latent_dim, bn)
    model = nn.Sequential(
        layers.ConvLayer(in_channel,        conv_channel,       filter_size, stride=2, padding = 2, bn=bn),
        layers.ConvLayer(conv_channel,      inner_conv_channel, filter_size, stride=2, padding = 2, bn=bn),
        layers.ConvLayer(inner_conv_channel,inner_conv_channel, filter_size, stride=1, padding = 2, bn=bn),
        layers.Flatten(),
        layers.Dense(flatten_img_size, dense_size),
        layers.Dense(dense_size,       latent_dim)
    )
    model = model.to(device=device, dtype=dtype)
    model = torch.nn.DataParallel(model, device_ids = GPU_IDs)
    return model
def Classifier(input_dim, dense_size, s_classes, bn):
    
    model = nn.Sequential(
        layers.Dense(input_dim,  dense_size, bn=bn),
        layers.Dense(dense_size, dense_size, bn=bn),
        layers.Dense(dense_size, s_classes,  bn=bn)
    )
    model = model.to(device=device, dtype=dtype)
    model = torch.nn.DataParallel(model, device_ids = GPU_IDs)
    return model
def Decoder(s_dim, z_dim, img_size, img_channel, conv_channel, filter_size, dense_size, bn):
    # TODO
    # essentially the mirror version of Encoder
    inner_conv_channel = conv_channel//2
    back_img_size = img_size//4
    flatten_img_size = inner_conv_channel * back_img_size * back_img_size
    
    input_dim = s_dim + z_dim
    
    pad = int(np.floor(filter_size/2)) # chose pad this way to fullfill floor((-F+2P)/1+1)==0
    
    model = nn.Sequential(
        layers.Dense(input_dim, dense_size),
        layers.Dense(dense_size, inner_conv_channel*back_img_size*back_img_size),
        layers.Reshape((-1, inner_conv_channel, back_img_size, back_img_size)),
        layers.ConvLayer(inner_conv_channel,    inner_conv_channel, filter_size, stride=1, padding=pad, bn=bn, upsampling=True),
        layers.ConvLayer(inner_conv_channel,    conv_channel, filter_size, stride=1, padding=pad, bn=bn, upsampling=True),
        layers.ConvLayer(conv_channel,    img_channel,       filter_size, stride=1, padding=pad, bn=bn, upsampling=False),
    )
    model = model.to(device=device, dtype=dtype)
    model = torch.nn.DataParallel(model, device_ids = GPU_IDs)
    return model
def AdvLayer(input_dim, dense_size, s_classes, bn):
    # same structure as Classifier
    return Classifier(input_dim, dense_size, s_classes, bn)

def S_Encoder(params):
    conv_channel= params.enc_conv_channel
    filter_size = params.enc_conv_filter_size
    img_size = params.img_size
    in_channel  = params.img_channel
    dense_size  = params.encdec_dense_size
    bn = params.s_enc_bn
    latent_dim = params.s_enc_dim
    
    model = Encoder(img_size, in_channel, conv_channel, filter_size, latent_dim, dense_size, bn)
    model.m_name='s_enc'
    return model
def Z_Encoder(params):
    conv_channel= params.enc_conv_channel
    filter_size = params.enc_conv_filter_size
    img_size = params.img_size
    in_channel  = params.img_channel
    dense_size  = params.encdec_dense_size
    bn = params.z_enc_bn
    latent_dim = params.z_enc_dim
    
    model = Encoder(img_size, in_channel, conv_channel, filter_size, latent_dim, dense_size, bn)
    model.m_name='z_enc'
    return model
def S_Classifier(params):
    input_dim   = params.s_enc_dim
    dense_size  = params.classifier_dense_size
    classes_num = params.classes_num
    bn          = params.classifier_use_bn
    
    model = Classifier(input_dim, dense_size, classes_num, bn)
    model.m_name='s_classifier'
    return model
def Z_AdvLayer(params):
    input_dim   = params.z_enc_dim
    dense_size  = params.classifier_dense_size
    classes_num = params.classes_num
    bn          = params.classifier_use_bn
    
    model = AdvLayer(input_dim, dense_size, classes_num, bn)
    return model
def SZ_Decoder(params):
    s_dim = params.s_enc_dim
    z_dim = params.z_enc_dim
    img_size = params.img_size
    img_channel = params.img_channel
    conv_channel = params.dec_conv_channel
    filter_size  = params.dec_conv_filter_size
    dense_size  = params.encdec_dense_size
    bn           = params.dec_use_bn
    model = Decoder(s_dim, z_dim, img_size, img_channel, conv_channel, filter_size, dense_size, bn)
    model.m_name='sz_dec'
    return model

def test_Encoder(model,params):
    x = torch.zeros((64, params.img_channel, params.img_size, params.img_size), dtype=dtype)
    scores = model(x)
    print(scores.size())  # you should see [64, latent_dim]
    print()
def test_classifier(model,params):
    x = torch.zeros((64, params.s_enc_dim), dtype=dtype)
    scores = model(x)
    print(scores.size())  # should see [64,classes_num]
    print()
def test_Decoder(model,params):
    x = torch.zeros((64, params.s_enc_dim + params.z_enc_dim), dtype=dtype)
    scores = model(x)
    print(scores.size())  # should see [64,classes_num]
    print()
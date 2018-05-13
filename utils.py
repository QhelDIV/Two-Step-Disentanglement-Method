# some setup code
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn import init
from torch.autograd import Variable
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import datetime

from config import *
from params import *




class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset. 
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))
 
    def __len__(self):
        return self.num_samples
    
class dataLoader():
    def __init__(self, dset_name, dset_path='./datasets/'):
        if dset_name == 'MNIST':
            self.params=MNISTParameters()
            self.setup_MNIST()
        if dset_name == 'SPRITES':
            self.params=SPRITESParameters()
            self.setup_SPRITES()
        self.iter_per_epoch = self.NUM_TRAIN // self.batch_size
    def setup_SPRITES(self):
        self.NUM_TRAIN = 50000
        self.NUM_VAL = 5000
        self.NUM_TEST= 5000
    def setup_MNIST(self):
        self.NUM_TRAIN = 50000
        self.NUM_VAL = 5000
        self.NUM_TEST= 5000
        img_size = self.params.img_size
        img_channel = self.params.img_channel

        self.batch_size = self.params.batch_size
        
        img_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0,)*img_channel, (1,)*img_channel)
        ])

        mnist_train = dset.MNIST('./datasets/MNIST_data', train=True, download=False,
                                   transform=img_transform)
        self.loader_train = DataLoader(mnist_train, batch_size=self.batch_size,
                                  sampler=ChunkSampler(self.NUM_TRAIN, 0),num_workers=8)

        mnist_val = dset.MNIST('./datasets/MNIST_data', train=True, download=False,
                                   transform=img_transform)
        self.loader_val = DataLoader(mnist_val, batch_size=self.batch_size,
                                sampler=ChunkSampler(self.NUM_VAL, self.NUM_TRAIN),num_workers=8)

        mnist_test = dset.MNIST('./datasets/MNIST_data', train=False, download=False,
                                   transform=img_transform)
        self.loader_test = DataLoader(mnist_test, batch_size=self.batch_size,
                                sampler=ChunkSampler(self.NUM_TEST,0),num_workers=8)
        
        # group images by class name
        self.img_grouped = [[] for i in range(self.params.classes_num)]
        for it, (xbat,ybat) in enumerate(self.loader_test):
            for i in range(len(ybat)):
                x = xbat[i]
                y = ybat[i]
                self.img_grouped[y.item()].append( x.view(img_size ** 2) )
                
        self.imgs = self.loader_test.__iter__().next()[0].view(self.batch_size, img_channel*img_size*img_size).numpy().squeeze()
        
    def show_imgs(self):
        showed = [self.img_grouped[2][i] for i in range(20)]
        showed += [self.img_grouped[6][i] for i in range(20)]
        torch.stack(showed)
        utils.show_images(torch.stack(showed))

def print_info():
    print(torch.__version__)
    print('using device:', device)
    print('data type:', dtype)
    print('VERBOSE==',VERBOSE)

def show_images(images,params):
    img_size = params.img_size
    img_channel = params.img_channel
    
    images = torch.tensor(images)
    images = images.view([images.shape[0], img_channel, img_size, img_size])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = img_size

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        img = img.permute(1,2,0).numpy()
        if img.shape[2]==1:
            img = img.reshape(img.shape[0],img.shape[1])
        plt.imshow(img)
    return 
def count_params(model):
    """Count the number of parameters in the current computation graph """
    param_count = np.sum([np.prod(p.size()) for p in model.parameters()])
    return param_count
def preprocess_img(x):
    return 2 * x - 1.0

def deprocess_img(x):
    return (x + 1.0) / 2.0

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def get_time():
    now = datetime.datetime.now()
    return now.strftime('%Y_%m_%d__%H_%M_%S')

# if not list, make list
def make_list(obj):
    if isinstance(obj,list)==False:
        return [obj]
    return obj
    
def save_models(models, path = var_save_path, mode='time', mode_param = None):
    if mode=='time':
        suffix = get_time()
    elif mode=='iter':
        suffix = str(mode_param)
    elif mode=='param':
        suffix = str(mode_param)
    for model in make_list(models):
        torch.save(model.state_dict(),path+ model.m_name + suffix)
        #torch.save(model,path+ model.m_name + 'MODEL' + suffix)
def load_models(models, path = load_path, suffix=''):
    for model in make_list(models):
        loaded = torch.load(path + model.m_name + suffix)
        model.load_state_dict(loaded)
        #torch.load(path+model.m_name)
def tort(x):
    # this func should pass (0,0) and (1,1)
    return -1000*x**2+1001*x
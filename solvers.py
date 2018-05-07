import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.nn import init

import matplotlib.pyplot as plt

import params
from params import img_size,img_channel
from config import *
import utils

import layers

def get_optimizer(model, which = 'adam'):
    #optimizer = optim.Adam( filter(lambda p: p.requires_grad, model.parameters() ), lr=1e-4, betas=(0.5,0.999))
    if which=='adam':
        optimizer = optim.Adam( model.parameters(), lr=params.learning_rate, betas=params.adam_beta)
    return optimizer
def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        
def test_recon(model, dloader):
    img = torch.stack([dloader.img_grouped[i][0] for i in range(params.classes_num)])
    imgr=img.view(-1,img_channel,img_size,img_size)
    imgo = model(imgr.to(device=device,dtype=dtype))
    imgo = imgo.view(-1,img_size*img_size).detach().to(device=device_CPU,dtype=dtype)
    
    #print(img.shape,imgo.shape,torch.cat((img,imgo)).shape)
    utils.show_images(torch.cat((img,imgo)))
class ClassifierSolver():
    def __init__(self, s_enc, s_classifier, dloader):
        self.s_enc = s_enc
        self.s_classifier = s_classifier
        self.classifier = nn.Sequential(
            s_enc,
            s_classifier
        )
        self.dloader = dloader
        self.it_count = 1
    def train(self,epochs=1):
        print_every = 1000
        model = self.classifier
        model = model.to(device=device) # move the model parameters to device
        optimizer = get_optimizer(model)
        
        for epoch in range(epochs):
            for it, (x,y) in enumerate(self.dloader.loader_train):
                model.train()
                x = x.to(device = device, dtype=dtype)
                y = y.to(device = device, dtype=torch.long) # QUESTION!!

                scores = model(x)
                loss = F.cross_entropy(scores, y)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                if self.it_count % print_every == 0:
                    print('iteration %d, loss = %.4f' % (self.it_count,loss.item()))
                    self.test(validation=True)
                    print()
                self.it_count += 1
    def test(self,validation=False):
        model = self.classifier
        if validation==True:
            loader = self.dloader.loader_val
            print('Checking accuracy on validation set')
        else:
            loader = self.dloader.loader_test
            print('Checking accuracy on test set')   
        num_correct = 0
        num_samples = 0
        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
                y = y.to(device=device, dtype=torch.long)
                scores = model(x)
                _, preds = scores.max(1)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
            acc = float(num_correct) / num_samples
            print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        return acc
class DisAdvSolver():
    def __init__(self, s_enc, z_enc, sz_dec, z_adv, dloader):
        self.dloader  = dloader
        
        self.s_enc = s_enc
        self.z_enc = z_enc
        self.sz_dec = sz_dec
        self.z_adv = z_adv
        self.save_list = [s_enc, z_enc, sz_dec, z_adv]
        
        self.recon_w = params.recon_w
        self.adv_w = params.adv_w

        self.adv_net = layers.AdvNet(z_enc,z_adv)
        self.disent_net = layers.DisentNet(z_enc,sz_dec)
        self.z_classifier = ClassifierSolver(z_enc,z_adv, self.dloader)
        
        self.set_mode('init')
        self.adv_solver = get_optimizer(self.adv_net)
        self.disent_solver = get_optimizer(self.disent_net)

        self.test_recon_net = layers.ReconNet(s_enc,z_enc,sz_dec)

        self.show_every = 3000
        self.adv_disent_ratio = params.adv_disent_ratio

        self.train_log = { 'loss':[], 'adv_acc':[] }
        
        
        self.it_count = 0
        
    def set_mode(self, mode):
        if mode=='disentangle':
            layers.set_trainable(self.s_enc,False)
            layers.set_trainable(self.z_enc,True)
            layers.set_trainable(self.sz_dec,True)
            layers.set_trainable(self.z_adv,False)
        elif mode == 'adversarial':
            layers.set_trainable(self.s_enc,False)
            layers.set_trainable(self.z_enc,False)
            layers.set_trainable(self.sz_dec,False)
            layers.set_trainable(self.z_adv,True)
        elif mode == 'init':
            layers.set_trainable(self.s_enc,True)
            layers.set_trainable(self.z_enc,True)
            layers.set_trainable(self.sz_dec,True)
            layers.set_trainable(self.z_adv,True)
        else:
            print("Unknown mode")
    def show_switch_latent(self, show_range = params.classes_num):
        s_latent=[]
        z_latent=[]
        img_lists = []
        s_enc = self.s_enc
        z_enc = self.z_enc
        sz_dec = self.sz_dec
        
        for classi in range(show_range):
            img = self.dloader.img_grouped[classi][0].view(1,img_channel,img_size,img_size)
            img = img.to(device=device,dtype=dtype)
            
            s_latent.append( s_enc(img) )
            z_latent.append( z_enc(img) )
        for row in range( show_range ):
            for col in range( show_range ):
                latent = torch.cat((s_latent[col],z_latent[row]),dim=1)
                recon  = sz_dec(latent)
                img_lists.append(recon.view( img_size*img_size ) )
                
        utils.show_images(torch.stack(img_lists).detach().cpu().numpy())
                
    def show_interpolated(self, inter_step = 4, tuples=((2,18),(6,10))):
        inter_img1 = self.dloader.img_grouped[tuples[0][0]][tuples[0][1]].view(1,img_channel,img_size,img_size)
        inter_img2 = self.dloader.img_grouped[tuples[1][0]][tuples[1][1]].view(1,img_channel,img_size,img_size)
        inter_img1 = inter_img1.to(device=device,dtype=dtype)
        inter_img2 = inter_img2.to(device=device,dtype=dtype)
        
        s_enc = self.s_enc
        z_enc = self.z_enc
        sz_dec = self.sz_dec

        s_lat1 = s_enc(inter_img1)
        z_lat1 = z_enc(inter_img1)
        s_lat2 = s_enc(inter_img2)
        z_lat2 = z_enc(inter_img2)

        weights = np.arange(0,1,1/(inter_step-1))
        weights = np.append(weights,1.)
        weights = torch.tensor(weights)
        weights = weights.to(device=device,dtype=dtype)

        #print(z_lat1,z_lat2)

        img_lists = []
        for row_w in weights:
            for col_w in weights:
                s_latent =  (1-row_w) * s_lat1 + row_w * s_lat2
                z_latent =  (1-col_w) * z_lat1 + col_w * z_lat2
                latent = torch.cat((s_latent,z_latent),dim=1)
                recon  = sz_dec(latent)
                img_lists.append(recon.view( img_size*img_size ) )

        utils.show_images(torch.stack(img_lists).detach().cpu().numpy())
        
       
        
    def train(self, epochs=1):
        disent_recon_loss = 1000000.
        disent_adv_loss = 1000000.
        disent_loss = 1000000.
        adv_loss = 1000000.
        dloader = self.dloader
        
        for epoch in range(epochs):
            for it, (x,y) in enumerate(dloader.loader_train):

                x = x.to(device=device, dtype=dtype)
                y = y.to(device=device, dtype=torch.long)

                ############ train disentangle net
                if self.it_count%(self.adv_disent_ratio+1) == 0:
                    self.set_mode('disentangle')
                    self.disent_solver.zero_grad()

                    s_latent = self.s_enc(x)
                    x_reconstructed = self.disent_net(s_latent,x)
                    disent_recon_loss = F.mse_loss(x_reconstructed, x, size_average=True)

                    scores = self.adv_net(x)
                    disent_adv_loss   = F.cross_entropy(scores, y)

                    disent_loss = self.recon_w * disent_recon_loss + self.adv_w * disent_adv_loss

                    disent_loss.backward()
                    self.disent_solver.step()
                else:
                ############# train adv net
                    self.set_mode('adversarial')
                    self.adv_solver.zero_grad()

                    scores = self.adv_net(x)
                    adv_loss   = F.cross_entropy(scores, y)

                    adv_loss.backward()
                    self.adv_solver.step()

                if (self.it_count % self.show_every == 0):
                    print('Iter: {}, rencon_loss: {:.4}, adv_loss: {:.4}, disent_loss:{:.4}'.format(self.it_count, disent_recon_loss,disent_adv_loss, disent_loss))
                    print('adv classifier accuracy:')
                    adv_acc = self.z_classifier.test()
                    
                    self.train_log['loss'].append((disent_recon_loss, adv_loss, disent_loss))
                    self.train_log['adv_acc'].append( adv_acc )
                    
                    self.show_switch_latent(4)
                    self.show_interpolated(4)
                    test_recon(self.test_recon_net,self.dloader)
                    plt.show()
                    
                    utils.save_models(self.save_list, mode='iter', mode_param=self.it_count)
                    
                    print()
                self.it_count += 1

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.nn import init

import matplotlib.pyplot as plt

import copy

from params import *
from config import *
import utils

import layers
import nets

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

class Solver():
    def __init__(self,dloader,params):
        self.dloader = dloader
        self.params = params
    def get_optimizer(self, model, which = 'adam'):
        #optimizer = optim.Adam( filter(lambda p: p.requires_grad, model.parameters() ), lr=1e-4, betas=(0.5,0.999))
        if which=='adam':
            optimizer = optim.Adam( model.parameters(), lr=self.params.learning_rate, betas=self.params.adam_beta)
        elif which=='SGD':
            optimizer = optim.SGD( model.parameters(), lr = self.params.learning_rate, momentum=0.9)
        return optimizer

class ReconSolver(Solver):
    def __init__(self,model, dloader,params):
        super().__init__(dloader, params)
        self.model = model
    def test(self,classes_num=None):
        if classes_num is None:
            classes_num = self.params.show_classes_num
        img_size = self.params.img_size
        img_channel = self.params.img_channel
        
        img_list = []
        for i in range(classes_num):
            show_class = self.params.show_classes[i]
            img = self.dloader.img_grouped[show_class[0]][show_class[1]]
            img_list.append(img)
        img = torch.stack(img_list)
        img = img.to(device=device_CPU,dtype=dtype)
        
        imgr=img.view(-1,img_channel,img_size,img_size)
        imgo = self.model(imgr.to(device=device,dtype=dtype))
        imgo = imgo.view(-1,img_channel*img_size*img_size).detach().to(device=device_CPU,dtype=dtype)

        #print(img.shape,imgo.shape,torch.cat((img,imgo)).shape)
        utils.show_images(torch.cat((img,imgo)),self.params)
class ClassifierSolver(Solver):
    def __init__(self, s_enc, s_classifier, dloader, params):
        super().__init__(dloader, params)
        self.s_enc = s_enc
        self.s_classifier = s_classifier
        self.classifier = nn.Sequential(
            s_enc,
            s_classifier
        )
        self.dloader = dloader
        self.it_count = 1
    def train(self,epochs=1,freeze_enc = False, silence=False):
        print_every = 100
        model = self.classifier
        model = model.to(device=device) # move the model parameters to device
        if freeze_enc==True:
            optimizer = self.get_optimizer(self.s_classifier)
        else:
            optimizer = self.get_optimizer(model)
        
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

                if silence==False and self.it_count % print_every == 0:
                    acc = self.test(mode = 'validation', silence=True)
                    print('iteration %d, loss = %.4f, acc=%.2f' % (self.it_count,loss.item(),acc),end='\r')
                self.it_count += 1
    def predict(self, x):
        x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
        scores = self.classifier(x)
        _, preds = scores.max(1)
        return preds
    def test(self,mode='validation', silence=False):
        if mode=='validation':
            loader = self.dloader.loader_val
            if silence==False:
                print('Checking accuracy on validation set')
        elif mode=='test':
            loader = self.dloader.loader_test
            if silence==False:
                print('Checking accuracy on test set')
        else:
            print("ERROR: wrong mode!")
        num_correct = 0
        num_samples = 0
        error_count = np.zeros(672)
        error_dict = dict()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
                y = y.to(device=device, dtype=torch.long)
                preds = self.predict(x)
                num_correct += (preds == y).sum()
                
                num_samples += preds.size(0)
            acc = float(num_correct) / num_samples
            
            
            if silence==False:
                print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        return acc
class DisAdvSolver(Solver):
    def __init__(self, s_enc=None, z_enc=None, sz_dec=None, z_adv=None, dloader=None, params=None, loading=False, loadPath = var_save_path, loadSuffix = '', model_name = '', saving_while_training=False):
        super().__init__(dloader, params)
        
        self.dloader  = dloader
        self.params = params
        self.model_name = model_name
        self.saving_while_training = saving_while_training
        
        if loading==False:
            self.s_enc = s_enc
            self.z_enc = z_enc
            self.sz_dec = sz_dec
            self.z_adv = z_adv
        else:
            self.z_enc = nets.Z_Encoder(self.params)
            self.z_adv = nets.Z_AdvLayer(self.params)
            self.sz_dec= nets.SZ_Decoder(self.params)
            self.s_enc = nets.S_Encoder(self.params)
        if loading==True:
            self.load_model(loadPath,loadSuffix)
        
        self.recon_w = self.params.recon_w
        self.adv_w = self.params.adv_w

        self.adv_net = layers.AdvNet(self.z_enc,self.z_adv)
        self.disent_net = layers.DisentNet(self.z_enc,self.sz_dec)
        
        self.set_mode('init')
        self.adv_solver = self.get_optimizer(self.adv_net)
        self.disent_solver = self.get_optimizer(self.disent_net)
        
        self.z_enc_solver = self.get_optimizer(self.z_enc)
        self.sz_dec_solver = self.get_optimizer(self.sz_dec)
        self.z_adv_solver = self.get_optimizer(self.z_adv, which = 'SGD')

        self.reconSolver = ReconSolver(layers.ReconNet(self.s_enc,self.z_enc,self.sz_dec), self.dloader, self.params)

        self.show_every = 2000
        self.save_num = 1
        self.save_every = dloader.iter_per_epoch
        self.log_every = 100
        self.adv_disent_ratio = self.params.adv_disent_ratio

        self.train_log = { 'loss':[], 'adv_acc':[] }
        
        
        self.it_count = 1
    def set_mode(self, mode):
        if mode=='disentangle':
            layers.set_trainable(self.s_enc,False)
            layers.set_trainable(self.z_enc,True)
            layers.set_trainable(self.sz_dec,True)
            layers.set_trainable(self.z_adv,True)
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
    def show_switch_latent(self, show_range = None):
        if show_range is None:
            show_range = self.params.show_classes_num
        s_latent=[]
        z_latent=[]
        img_lists = []
        real_lists = []
        ground_truth_list = []
        s_enc = self.s_enc
        z_enc = self.z_enc
        sz_dec = self.sz_dec
        img_size = self.params.img_size
        img_channel = self.params.img_channel
        show_classes = self.params.show_classes
        
        # first image is a black one
        img_lists.append( torch.tensor( np.zeros(img_channel*img_size*img_size) ).to(device=device,dtype=dtype)  )
        
        for classi in range(show_range):
            show_class = show_classes[classi]
            img = self.dloader.img_grouped[show_class[0]][show_class[1]].view(1,img_channel,img_size,img_size)
            real_lists.append(self.dloader.img_grouped[show_class[0]][show_class[1]])
            img = img.to(device=device,dtype=dtype)
            
            ground_truth_list.append( img.view( img_channel*img_size*img_size ) )
            s_latent.append( s_enc(img) )
            z_latent.append( z_enc(img) )
        img_lists.extend( ground_truth_list )
            
        for row in range( show_range ):
            img_lists.append( ground_truth_list[row] )
            for col in range( show_range ):
                latent = torch.cat((s_latent[col],z_latent[row]),dim=1)
                recon  = sz_dec(latent)
                img_lists.append(recon.view( img_channel*img_size*img_size ) )
                
        utils.show_images(torch.stack(img_lists).detach().cpu().numpy(),self.params)
        utils.show_images(torch.stack(real_lists).detach().cpu().numpy(),self.params)
                
    def show_interpolated(self, inter_step = 4, tuples=None):
        if tuples is None:
            tuples = self.params.interpolated_tuples
        img_size = self.params.img_size
        img_channel = self.params.img_channel
        
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
                img_lists.append(recon.view( img_channel*img_size*img_size ) )

        utils.show_images(torch.stack(img_lists).detach().cpu().numpy(),self.params)
        
       
        
    def train(self, epochs=1):
        disent_recon_loss = 1000000.
        disent_adv_loss = 1000000.
        disent_loss = 1000000.
        adv_loss = 1000000.
        dloader = self.dloader
        self.max_it_count = dloader.iter_per_epoch * epochs
        self.cur_it_count = 0
        self.save_count = 0
        self.save_num = 6
        self.show_num = 1
        #self.save_every = dloader.iter_per_epoch * epochs // self.save_num
        self.save_every = 10000
        #self.show_every = dloader.iter_per_epoch * epochs // self.show_num
        self.show_every = 10000
        
        for epoch in range(epochs):
            for it, (x,y) in enumerate(dloader.loader_train):

                x = x.to(device=device, dtype=dtype)
                y = y.to(device=device, dtype=torch.long)
                if self.it_count==1:
                    first_x = x
                    first_y = y

                ############ train disentangle net
                if self.it_count%(self.adv_disent_ratio+1) == 0:
                    self.set_mode('disentangle')
                    self.z_enc_solver.zero_grad()
                    self.sz_dec_solver.zero_grad()

                    s_latent = self.s_enc(x)
                    z_latent = self.z_enc(x)

                    latent = torch.cat((s_latent,z_latent),dim=1)
                    reconstructed = self.sz_dec(latent)
                    
                    disent_recon_loss = F.mse_loss(reconstructed, x, size_average=True)

                    scores = self.z_adv(z_latent)
                    disent_adv_loss   = F.cross_entropy(scores, y)

                    disent_loss = self.recon_w * disent_recon_loss + self.adv_w * disent_adv_loss

                    disent_loss.backward()
                    self.z_enc_solver.step()
                    self.sz_dec_solver.step()
                else:
                ############# train adv net
                    self.set_mode('adversarial')
                    self.z_adv_solver.zero_grad()

                    z_latent = self.z_enc(x)
                    scores = self.z_adv(z_latent)
                    adv_loss   = F.cross_entropy(scores, y)

                    adv_loss.backward()
                    self.z_adv_solver.step()
                    
                if (self.it_count % self.log_every ==0):
                    self.train_log['loss'].append((disent_recon_loss, adv_loss, disent_loss))
                    print('progress: %.2f, total iter: %d, losses: recon:%.4f, disent:%.4f, adv:%.4f ' % (self.cur_it_count/self.max_it_count, self.it_count, disent_recon_loss, disent_loss, adv_loss), end= '\r')
                    
                if (self.it_count % self.show_every==0):
                    print()
                    self.show_switch_latent(10)
                    self.show_interpolated(10)
                    self.reconSolver.test()
                    plt.show()
                    print()
                    
                if ((self.saving_while_training==True) and (self.it_count%self.save_every==0)):
                    self.save_count += 1
                    self.save_model(suffix = self.model_name+'_num'+str(self.it_count))
                    print()
                # it_count is total accumulated count, while cur_it_count is count in this training session
                self.it_count += 1
                self.cur_it_count += 1

        plt.show()
    def test_disentangle(self):
        # S_Classifier and Z_AdvLayer are same, except input dimension...
        print('training a classifier on top of z encoder...')
        self.z_adv = nets.Z_AdvLayer(self.params)
        z_enc_clone = copy.deepcopy(self.z_enc)
        z_adv_clone = copy.deepcopy(self.z_adv)
        z_enc_tester = ClassifierSolver(z_enc_clone, z_adv_clone ,self.dloader, self.params)
        z_enc_tester.train(1,freeze_enc=True,silence=True)
        print('testing this classifier...')
        z_enc_tester.test(mode='test')

    def plot_history(self):
        print('reconstruction loss curve')
        loss_hist = self.train_log['loss']
        recon_loss_hist = [tup[0] for tup in loss_hist]
        plt.plot(recon_loss_hist)
        plt.show()
    def save_model(self,path = var_save_path, suffix = None):
        if suffix is None:
            suffix = self.model_name
        save_list = [self.s_enc, self.z_enc, self.sz_dec, self.z_adv]
        save_list[0].m_name = 's_enc';         save_list[1].m_name = 'z_enc';
        save_list[2].m_name = 'sz_dec';        save_list[3].m_name = 'z_adv';
        utils.save_models(save_list,path, mode='param', mode_param = suffix)
    def load_model(self, path = var_save_path, suffix = ''):
        load_list = [self.s_enc, self.z_enc, self.sz_dec, self.z_adv]
        load_list[0].m_name = 's_enc';         load_list[1].m_name = 'z_enc';
        load_list[2].m_name = 'sz_dec';        load_list[3].m_name = 'z_adv';
        
        utils.load_models(load_list,path = path, suffix = suffix)

class HPTuner():
    def __init__(self,s_enc, dloader, params):
        
        self.combinations = params.hyperparam_combinations()
        self.epoch_num = 100
        self.index = 0
        self.max_index = len(self.combinations)
        self.dloader = dloader
        self.params = params
        self.s_enc = s_enc
        
    def tune(self, epoch_num=100):
        
        self.epoch_num = epoch_num
        is_saving = False
        save_list = []
        try:
        
            print('learning_rate, hlayer_size, training_epochs_num, reg_strengths')
            while True:
                hyperparameters = self.combinations[self.index]
                print('training in this scheme:','{}/{}'.format(self.index+1,self.max_index))
                print(hyperparameters)
                print()

                z_enc = nets.Z_Encoder(self.params)
                z_adv = nets.Z_AdvLayer(self.params)
                sz_dec= nets.SZ_Decoder(self.params)
                s_enc = self.s_enc
                solver = DisAdvSolver(s_enc, z_enc, sz_dec, z_adv, self.dloader, self.params)
                solver.show_every = self.dloader.iter_per_epoch * self.epoch_num

                solver.train(self.epoch_num)
                solver.plot_history()
                
                print('saving model...')
                is_saving = True
                solver.save_model(suffix = str(hyperparameters))
                is_saving = False
                
                self.index += 1
                if self.index == self.max_index:
                    break
        except KeyboardInterrupt:
            print('KeyboardInterrupt! you can keep going by rerun the cell, it will continue from where it was interrupted')
            if is_saving == True:
                print('resume saving...')
                solver.save_model(suffix = str(hyperparameters))
            
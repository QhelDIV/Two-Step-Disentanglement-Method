import itertools
# config file for Sprites dataset
# general config
class Parameters:
    def __init__(self): 
        ####### hyperparameters candidates #######
        self.learning_rates = [5e-3,1e-3,5e-4]
        self.adam_betas = [(0.9,0.999)]
        self.adv_ws = [-0.1,-0.05]
        self.adv_disent_ratios = [3,4,5]

        ####### SYS CONFIG ########
        self.batch_size = 256
        # learning settings
        self.learning_rate =.001
        self.adam_beta = (0.9,0.999)
        self.DATASET_NAME = 'UNDETERMINT'
        #DATASET_NAME = 'SPRITES'

        self.adv_disent_ratio = 6
        
        self.recon_w  = 1
        self.adv_w   = -0.1
        
        # enc, dec config
        self.encdec_dense_size   = 256

        # Encoder config
        self.enc_conv_filter_size = 5
        self.enc_conv_channel    = 16
        self.enc_use_bn        = True

        self.s_enc_bn = False
        self.s_enc_dim= 16

        self.z_enc_bn = True
        self.z_enc_dim= 16

        # classifier
        self.classifier_dense_size = 256
        self.classes_num = 10
        self.classifier_use_bn = True

        # Decoder config
        self.dec_conv_filter_size = 5
        self.dec_conv_channel    = 16
        self.dec_use_bn        = True

        # Adv net config
        # same with s classifier
        # see DisAdvNet.ipyn
    def hyperparam_combinations(self):
        hyperpList=[self.learning_rates, self.adam_betas, self.adv_ws, self.adv_disent_ratios]
        return list(itertools.product(*hyperpList))

class MNISTParameters(Parameters):
    def __init__(self):
        super().__init__()
        self.DATASET_NAME = 'MNIST'
        self.img_size    = 28
        self.img_channel = 1
class SPRITESParameters(Parameters):
    def __init__(self):
        super().__init__()
        self.DATASET_NAME = 'SPRITES'
        self.img_size    = 60
        self.img_channel = 3
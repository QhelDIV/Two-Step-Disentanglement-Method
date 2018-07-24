import itertools
# config file for Sprites dataset
# general config
class Parameters:
    def __init__(self): 
        ####### hyperparameters candidates #######
        self.learning_rates = [1e-3]
        self.adam_betas = [(0.9,0.999)]
        self.adv_ws = [-2]
        self.adv_disent_ratios = [4]

        ####### SYS CONFIG ########
        self.batch_size = 256 
        # learning settings
        self.learning_rate =1e-3
        self.adam_beta = (0.9,0.999)
        self.DATASET_NAME = 'UNDETERMINT'
        #DATASET_NAME = 'SPRITES'

        self.adv_disent_ratio = 4
        
        self.recon_w  = 1
        self.adv_w   = -0.5
        
        # enc, dec config
        self.encdec_dense_size   = 256

        # Encoder config
        self.enc_conv_filter_size = 5
        self.enc_conv_channel    = 16
        self.enc_use_bn        = True

        self.s_enc_bn = False
        self.s_enc_dim= 16

        self.z_enc_bn = False
        self.z_enc_dim= 16

        # classifier
        self.classifier_dense_size = 256
        self.classifier_use_bn = True

        # Decoder config
        self.dec_conv_filter_size = 5
        self.dec_conv_channel    = 16
        self.dec_use_bn        = False

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
        self.classes_num = 10
        self.interpolated_tuples = ((2,0),(6,0))
class SPRITESParameters(Parameters):
    def __init__(self):
        super().__init__()
        self.DATASET_NAME = 'SPRITES'
        self.img_size    = 32
        self.img_channel = 3
        self.classes_num = 336
        self.batch_size = 256
        
        self.show_classes_num = 10
        self.show_classes = [(1,0),(4,4),(50,3),(100,3),(160,3),(65,3),(210,0),(235,1),(330,8),(300,0)]
        
        self.interpolated_tuples = ((59,2),(332,2))
        
        self.s_enc_dim= 32
        self.z_enc_dim= 128
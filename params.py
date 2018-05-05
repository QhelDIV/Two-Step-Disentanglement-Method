# config file for MNIST dataset

####### SYS CONFIG ########
learning_rate = 1e-4
adv_disent_ratio = 4

recon_w  = 5
adv_w   = -10

# general config
img_size    = 28
img_channel = 1

# enc, dec config
encdec_dense_size   = 256

# Encoder config
enc_conv_filter_size = 5
enc_conv_channel    = 16
enc_use_bn        = True

s_enc_bn = False
s_enc_dim= 16

z_enc_bn = False
z_enc_dim= 16

# classifier
classifier_dense_size = 256
classes_num = 10
classifier_use_bn = True

# Decoder config
dec_conv_filter_size = 5
dec_conv_channel    = 16
dec_use_bn        = True

# Adv net config
# same with s classifier
# see DisAdvNet.ipyn

import torch

USE_GPU = True
dtype = torch.float32 # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
    GPU_IDs = [0,1,2,3]
else:
    device = torch.device('cpu')
    if USE_GPU == True:
        print("WARNING: cuda is unavailable, use CPU instead...")
device_CPU = torch.device('cpu')

VERBOSE = False
#VERBOSE = True

var_save_path = 'saved_models/var/'
save_path    = 'saved_models/'
load_path    = 'saved_models/'

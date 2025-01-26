import torch
import numpy as np

if torch.cuda.is_available():                               #If cuda is available, use cuda
    torch_device = 'cuda'
    float_dtype = np.float32
    torch.set_default_tensor_type(torch.cuda.FloatTensor)   #Use torch.cuda as default tensor
else:                                                       #If not, use cpu
    torch_device = 'cpu'
    float_dtype = np.float32
    torch.set_default_tensor_type(torch.DoubleTensor)       #Using vanilla torch as default

def torch_mod(x):
    return torch.remainder(x, 2*np.pi)

def torch_wrap(x):
    return torch_mod(x + np.pi) - np.pi

def grab(var):
    return var.detach().cpu().numpy()
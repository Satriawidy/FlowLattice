import torch
import packaging.version
if packaging.version.parse(torch.__version__) < packaging.version.parse('1.5.0'):
    raise RuntimeError('Torch version lower than 1.5.0 not supported')

def make_conv_net(*, hidden_sizes, kernel_size, in_channels, out_channels, use_final_tanh):
    sizes = [in_channels] + hidden_sizes + [out_channels] #list of number of configs in each layer
    assert packaging.version.parse(torch.__version__) >= packaging.version.parse('1.5.0')
    assert kernel_size % 2 == 1, 'kernel size must be odd for PyTorch >= 1.5.0'

    padding_size = (kernel_size // 2)  #To obtain same dim in both direction, p = k // 2
    net = []
    for i in range(len(sizes) - 1):
        net.append(torch.nn.Conv2d(
            sizes[i], sizes[i + 1], kernel_size, padding=padding_size,
            stride = 1, padding_mode='circular'
        ))                                      #Use convolution to transform from a layer to next layer
        if i != len(sizes) - 2:
            net.append(torch.nn.LeakyReLU())    #Use Leaky ReLU as activation function for all layers
        else:
            if use_final_tanh:
                net.append(torch.nn.Tanh())     #Use tanh as activation function for final layer

    return torch.nn.Sequential(*net)            

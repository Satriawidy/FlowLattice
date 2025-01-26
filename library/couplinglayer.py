from .initial import *
from .masking import *
from .conv import *
from .theory import *

def set_weights(m):
    if hasattr(m, 'weight') and m.weight is not None:
        torch.nn.init.normal_(m.weight, mean=1, std=2)
    if hasattr(m, 'bias') and m.bias is not None:
        m.bias.data.fill_(-1)

#Create class for simple normal distribution, similar to torch.distribution object
class SimpleNormal:
    def __init__(self, loc, var):                               #loc gives size of config, while var gives
        self.dist = torch.distributions.normal.Normal(           #the max scale for the distrib value
            torch.flatten(loc), torch.flatten(var))
        self.shape = loc.shape
    def log_prob(self, x):                                      #x gives n number of configs
        logp = self.dist.log_prob(x.reshape(x.shape[0], -1))    #compute log_prob for each config
        return torch.sum(logp, dim=1)                           #return sum of log_prob for each config
    def sample_n(self, batch_size):                             #batch_size gives the number of sample configs
        x = self.dist.sample((batch_size, ))                    #sample n (batch_size) configs 
        return x.reshape(batch_size, *self.shape)               #reshape to n configs

class SimpleCouplingLayer(torch.nn.Module):                     #Class for simple coupling layer 
    def __init__(self):                                         #Won't be used most of the time
        super().__init__()
        self.s = torch.nn.Sequential(
            torch.nn.Linear(1, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
            torch.nn.Tanh()
        )
    
    def forward(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        s = self.s(x2.unsqueeze(-1)).squeeze(-1)
        fx1 = torch.exp(s) * x1
        fx2 = x2
        logJ = s
        return torch.stack((fx1, fx2), dim=-1), logJ
    
    def reverse(self, fx):
        fx1, fx2 = fx[:, 0], fx[:, 1]
        x2 = fx2
        s = self.s(x2.unsqueeze(-1)).squeeze(-1)
        logJ = -s
        x1 = torch.exp(-s) * fx1
        return torch.stack((x1, x2), dim=-1), logJ

class AffineCoupling(torch.nn.Module):                          #Class for affine coupling layer (scalar phi4)
    def __init__(self, net, *, mask_shape, mask_parity):        #Accommodate masking (similar with simple coupling layer)
        super().__init__()                                      #Inherit functionality from torch.nn.Module
        self.mask = make_checker_mask(mask_shape, mask_parity)  #Create the masking depending on parity = 0,1
        self.net = net                                          #Use input CNN to aggregate
    
    def forward(self, x):                                       #Forward direction of the transformation
        x_frozen = self.mask * x                                #Masked is frozen
        x_active = (1 - self.mask) * x                          #Unmasked is active
        net_out = self.net(x_frozen.unsqueeze(1))               #CNN-Aggregate using frozen x
        s, t = net_out[:, 0], net_out[:, 1]                     #Half of CNN output is s, another half is t
        fx = (1 - self.mask) * t + x_active * torch.exp(s) + x_frozen #Here is the formula, below
                                                                #xf' = xf, xa' = xa * exp(s(xf)) + t(xf)
        axes = range(1, len(s.size()))
        logJ = torch.sum((1 - self.mask) * s, dim = tuple(axes))#logJf = 0, logJa = sum(s(xf))
        return fx, logJ
    
    def reverse(self, fx):
        fx_frozen = self.mask * fx
        fx_active = (1 - self.mask) * fx
        net_out = self.net(fx_frozen.unsqueeze(1))              #Will still give s(xf), t(xf) as xf' = xf
        s, t = net_out[:, 0], net_out[:, 1]
        x = (fx_active - (1 - self.mask) * t) * torch.exp(-s) + fx_frozen  
                                                                #xf = xf', xa = exp(-s(xf')) * (xa' - t(xf'))
        axes = range(1, len(s.size()))
        logJ = torch.sum((1 - self.mask) * (-s), dim = tuple(axes))
        return fx, logJ

def make_phi4_affine_layers(*, n_layers, lattice_shape, hidden_sizes, kernel_size): #Assembling layers for phi4
    layers = []
    for i in range(n_layers):
        parity = i % 2                              #Change parity in each turn of coupling layer
        net = make_conv_net(                        #Construct the CNN for s and t
            in_channels = 1, out_channels = 2, hidden_sizes = hidden_sizes,
            kernel_size = kernel_size, use_final_tanh = True)
        coupling = AffineCoupling(net, mask_shape = lattice_shape, mask_parity = parity)    #Construct the layer
        layers.append(coupling)                     #Assemble the layer
    return torch.nn.ModuleList(layers)

def apply_flow_to_prior(prior, coupling_layers, *, batch_size):     #Applying all layers to prior distribution
    x = prior.sample_n(batch_size)
    logq = prior.log_prob(x)
    for layer in coupling_layers:
        x, logJ = layer.forward(x)
        logq = logq - logJ
    return x, logq

#-------------------U1--------------------
class MultivariateUniform(torch.nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.dist = torch.distributions.uniform.Uniform(a, b)
    
    def log_prob(self, x):
        axes = range(1, len(x.shape))
        return torch.sum(self.dist.log_prob(x), dim = tuple(axes))
    
    def sample_n(self, batch_size):
        return self.dist.sample((batch_size, ))

class GaugeEquivCouplingLayer(torch.nn.Module):
    def __init__(self, *, lattice_shape, mask_mu, mask_off, plaq_coupling):
        super().__init__()
        link_mask_shape = (len(lattice_shape), ) + lattice_shape
        self.active_mask = make_2d_link_active_stripes(link_mask_shape, mask_mu, mask_off)
        self.plaq_coupling = plaq_coupling
    
    def forward(self, x):
        plaq = compute_u1_plaq(x, mu = 0, nu = 1)
        new_plaq, logJ = self.plaq_coupling(plaq)
        delta_plaq = new_plaq - plaq
        delta_links = torch.stack((delta_plaq, -delta_plaq), dim=1) # signs for U vs Udagger
        fx = self.active_mask * torch_mod(delta_links + x) + (1 - self.active_mask) * x
        return fx, logJ
    
    def reverse(self, fx):
        new_plaq = compute_u1_plaq(fx, mu=0, nu=1)
        plaq, logJ = self.plaq_coupling.reverse(new_plaq)
        delta_plaq = plaq - new_plaq
        delta_links = torch.stack((delta_plaq, -delta_plaq), dim=1) # signs for U vs Udagger
        x = self.active_mask * torch_mod(delta_links + fx) + (1 - self.active_mask) * fx
        return x, logJ

def tan_transform(x, s):
    return torch_mod(2 * torch.atan(torch.exp(s) * torch.tan(x/2)))

def tan_transform_logJ(x, s):
    return -torch.log(torch.exp(-s) * torch.cos(x/2)**2 + torch.exp(s) * torch.sin(x/2)**2)

def mixture_tan_transform(x, s):
    assert len(x.shape) == len(s.shape), \
        f'Dimension mismatch between x and s {x.shape} vs {s.shape}'
    return torch.mean(tan_transform(x, s), dim=1, keepdim=True)

def mixture_tan_transform_logJ(x, s):
    assert len(x.shape) == len(s.shape), \
        f'Dimension mismatch between x and s {x.shape} vs {s.shape}'
    return torch.logsumexp(tan_transform_logJ(x, s), dim=1) - np.log(s.shape[1])

def invert_transform_bisect(y, *, f, tol, max_iter, a=0, b = 2*np.pi):
    min_x = a * torch.ones_like(y)
    max_x = b * torch.ones_like(y)
    min_val = f(min_x)
    max_val = f(max_x)

    with torch.no_grad():
        for i in range(max_iter):
            mid_x = (min_x + max_x) / 2
            mid_val = f(mid_x)
            greater_mask = (y > mid_val).int()
            greater_mask = greater_mask.float()
            err = torch.max(torch.abs(y - mid_val))
            if err < tol: return mid_x
            if torch.all(mid_x == min_x) + (mid_x == max_x):
                print('WARNING: Reached floating point precision before tolerance'
                      f'(iter {i}, err {err})')
                return mid_x
            min_x = greater_mask * mid_x + (1 - greater_mask) * min_x
            min_val = greater_mask * mid_val + (1 - greater_mask) * min_val
            max_x = (1 - greater_mask) * mid_val + greater_mask * max_val
        print(f'WARNING: Did not converge to tol {tol} in {max_iter} iters! Error was {err}')
        return mid_x

def stack_cos_sin(x):
    return torch.stack((torch.cos(x), torch.sin(x)), dim=1)

class NCPPlaqCouplingLayer(torch.nn.Module):
    def __init__(self, net, *, mask_shape, mask_mu, mask_off,
                 inv_prec = 1e-6, inv_max_iter = 1000):
        super().__init__()
        assert len(mask_shape) == 2, (
            f'NCPPlaqCouplingLayer is implemented only in 2D, '
            f'mask_shape {mask_shape} is invalid'
        )
        self.mask = make_plaq_masks(mask_shape, mask_mu, mask_off)
        self.net = net
        self.inv_prec = inv_prec
        self.inv_max_iter = inv_max_iter
    
    def forward(self, x):
        x2 = self.mask['frozen'] * x
        net_out = self.net(stack_cos_sin(x2))
        assert net_out.shape[1] >= 2, 'CNN must output n_mix (s_i) + 1 (t) channels'
        s, t = net_out[:, :-1], net_out[:, -1]

        x1 = self.mask['active'] * x
        x1 = x1.unsqueeze(1)
        local_logJ = self.mask['active'] * mixture_tan_transform_logJ(x1, s)
        axes = tuple(range(1, len(local_logJ.shape)))
        logJ = torch.sum(local_logJ, dim=axes)
        fx1 = self.mask['active'] * mixture_tan_transform(x1, s).squeeze(1)

        fx = (
            self.mask['active'] * torch_mod(fx1 + t) +
            self.mask['passive'] * x +
            self.mask['frozen'] * x
        )
        return fx, logJ
    
    def reverse(self, fx):
        fx2 = self.mask['frozen'] * fx
        net_out = self.net(stack_cos_sin(fx2))
        assert net_out.shape[1] >= 2, 'CNN must output n_mix (s_i) + 1 (t) channels'
        s, t = net_out[:, :-1], net_out[:, -1]

        x1 = torch_mod(self.mask['active'] * (fx - t).unsqeeze(1))
        transform = lambda x: self.mask['active'] * mixture_tan_transform(x, s)
        x1 = invert_transform_bisect(
            x1, f=transform, tol=self.inv_prec, max_iter=self.inv_max_iter
        )
        local_logJ = self.mask['active'] * mixture_tan_transform_logJ(x1, s)
        axes = tuple(range(1, len(local_logJ.shape)))
        logJ = -torch.sum(local_logJ, dim=axes)
        x1 = x1.squeeze(1)

        x = (
            self.mask['active'] * x1 +
            self.mask['passive'] * fx +
            self.mask['frozen'] * fx2
        )
        return x, logJ
    
def make_u1_equiv_layers(*, n_layers, n_mixture_comps, lattice_shape, hidden_sizes, kernel_size):
    layers = []
    for i in range(n_layers):
        #periodically loop through all arrangements of maskings
        mu = i % 2
        off = (i // 2) % 4
        in_channels = 2 # x-> (cos(x), sin(x))
        out_channels = n_mixture_comps + 1 # for mixture s and t, respectively
        net = make_conv_net(in_channels= in_channels, out_channels=out_channels,
                            hidden_sizes=hidden_sizes, kernel_size=kernel_size,
                            use_final_tanh=False)
        plaq_coupling = NCPPlaqCouplingLayer(
            net, mask_shape=lattice_shape, mask_mu=mu, mask_off=off
        )
        link_coupling = GaugeEquivCouplingLayer(
            lattice_shape=lattice_shape, mask_mu=mu, mask_off=off,
            plaq_coupling=plaq_coupling
        )
        layers.append(link_coupling)
    return torch.nn.ModuleList(layers)
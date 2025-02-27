from .initial import torch_wrap, grab
import torch
import numpy as np

def bootstrap(x, *, Nboot, binsize):
    boots = []
    x = x.reshape(-1, binsize, *x.shape[1:])
    for i in range(Nboot):
        boots.append(np.mean(x[np.random.randint(len(x), size = len(x))], axis = (0,1)))
    return np.mean(boots), np.std(boots)

#-----------------Scalar-------------------

class ScalarPhi4Action:
    def __init__(self, M2, lam):
        self.M2 = M2
        self.lam = lam
    def __call__(self, cfgs):
        action_density = self.M2 * (cfgs ** 2) + self.lam * (cfgs ** 4)
        Nd = len(cfgs.shape) - 1
        dims = range(1, Nd + 1)
        for mu in dims:
            action_density += 2 * (cfgs ** 2)
            action_density -= cfgs * torch.roll(cfgs, -1, mu)
            action_density -= cfgs * torch.roll(cfgs, 1, mu)
        return torch.sum(action_density, dim = tuple(dims))

def twopointfuncs(cfgs):
    L = len(cfgs[0])                                                    #Lattice length
    C = 0                                                               #configurations with all entries equal zero
    for mu in range(L):                                                 #For every 0 direction
        for nu in range(L):                                             #For every 1 direction
            C = C + cfgs * np.roll(cfgs, (mu, nu), axis = (1, 2))       #Equivalent to G(x) = sum_y phi(y) * phi(y + x)
    return C

def susceptibility(cfgs):
    C = twopointfuncs(cfgs)
    X = np.mean(C, axis = (1,2))                                        #Average over all direction of G(x)
    return X

def zerogreenfuncs(cfgs):
    L = len(cfgs[0])                                                    #Lattice length
    D = []
    for mu in range(L):                                                 #For every t direction
        E = 0                                                           #Create a container
        for nu in range(L):                                             #For every x direction
            E += cfgs * np.roll(cfgs, (mu, nu), axis = (1, 2))          #Equivalent to sum_vec(x) phi(y) * phi(y + x)
        D.append(np.mean(E, axis = (1, 2)))                             #Equivalent to sum_y sum_vec(x) phi(y) * phi(y + x)
    
    X = np.stack(D, axis = 0).T / L                                     #Stack and reshape
    Y = (np.roll(X, - 1, axis = 1)[:,1:L-1] + np.roll(X, 1, axis = 1)[:,1:L-1]) / (2 * X[:,1:L-1])
    Y[Y < 1] = 1

    return X, np.arccosh(Y)                                             #Return Gtilde(0, t) and mp_eff

def isingenergy(cfgs):
    X = cfgs * (np.roll(cfgs, 1, axis = 1) + np.roll(cfgs, 1, axis = 2)) / 2
    return np.mean(X, axis = (1, 2))

def variablegenerator(flowphi4, num_total, num_cfgs, num_therm, L):
    flowphi4.run(train=False)
    Gp = []
    mp = []
    X = []
    Y = []
    for i in range(num_total):
        phi4_ens = flowphi4.make_mcmc_ensemble(N_samples = num_cfgs)
        print(f"Accept rate for (L = {L}):", np.mean(phi4_ens['accepted']))
        cfgs = np.array(phi4_ens['x'])[num_therm:]
        Gp0, mp0 = zerogreenfuncs(cfgs)
        Gp.append(Gp0)
        mp.append(mp0)
        X.append(susceptibility(cfgs))
        Y.append(isingenergy(cfgs))
    return np.array(Gp0), np.array(mp0), np.array(X), np.array(Y)



#---------------------U1-------------------

def compute_u1_plaq(links, mu, nu):
    return (links[:, mu] + torch.roll(links[:, nu], -1, mu + 1)
            - torch.roll(links[:, mu], -1, nu + 1) - links[:, nu])

def topo_charge(x):
    P01 = torch_wrap(compute_u1_plaq(x, mu = 0, nu = 1))
    axes = tuple(range(1, len(P01.shape)))
    return torch.sum(P01, dim = axes) / (2 * np.pi)

class U1GaugeAction:
    def __init__(self, beta):
        self.beta = beta
    def __call__(self, cfgs):
        Nd = cfgs.shape[1]
        action_density = 0
        for mu in range(Nd):
            for nu in range(mu + 1, Nd):
                action_density = action_density + torch.cos(compute_u1_plaq(cfgs, mu, nu))
        return -self.beta * torch.sum(action_density, dim = tuple(range(1, Nd + 1)))


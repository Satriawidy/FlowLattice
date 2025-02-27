from .initial import torch_device, grab
from .couplinglayer import apply_flow_to_prior
from .plot import init_live_plot, update_plots
from tqdm import tqdm
import base64
import io
import torch
import numpy as np
import matplotlib.pyplot as plt

def calc_dkl(logp, logq):           #Calculating KL divergence between model distribution q
    return (logq - logp).mean()     #and target distribution p

def compute_ess(logp, logq):        #Calculating effective sample size (ess) as validation metric
    logw = logp - logq              #log(p_i/q_i)
    log_ess = 2 * torch.logsumexp(logw, dim = 0) - torch.logsumexp(2 * logw, dim = 0) 
                                    #log(sum(p_i/q_i)^2 / sum((p_i/q_i)^2))
    ess_per_cfg = torch.exp(log_ess) / len(logw) #ESS / N
    return ess_per_cfg

def train_step(model, action, loss_fn, optimizer, metrics, batch_size):     #Training the model
    layers, prior = model['layers'], model['prior']                         #Obtaining the coupling layers and prior             
    optimizer.zero_grad()                                                   #Zero-ing the optimizer

    x, logq = apply_flow_to_prior(prior, layers, batch_size = batch_size)   #Apply the flow, obtain model distribution
    logp = - action(x)                                                      #Compute target distribution
    loss = loss_fn(logp, logq)                                              #Compute the loss
    loss.backward()                                                         #Compute the backpropagation of the loss

    optimizer.step()                                                        #Perform gradient descent

    metrics['loss'].append(grab(loss))                                      #Appending the training step results into
    metrics['logp'].append(grab(logp))                                      #the list
    metrics['logq'].append(grab(logq))
    metrics['ess'].append(grab(compute_ess(logp, logq)))

def print_metrics(history, avg_last_N_epochs, era, epoch):                  #Function to print the training results
    print(f' == Era {era} | Epoch {epoch} metrics')                         #for each era (some number of epochs)
    for key, val in history.items():
        avgd = np.mean(val[-avg_last_N_epochs:])
        print(f'\t{key} {avgd:g}')

class DiscreteFlow():
    def __init__(self, model, action, loss_fn, optimizer, batch_size, N_era, N_epoch, 
                 print_freq = None, plot_freq = None, use_pretrained = False, blob = None):
        self.model = model
        self.action = action
        self.criterion = loss_fn
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.N_era = N_era
        self.N_epoch = N_epoch
        self.use_pretrained = use_pretrained
        self.history = {
            'loss' : [],
            'logp' : [],
            'logq' : [],
            'ess' : []
        }
        if print_freq != None:
            self.print_freq = print_freq
        else:
            self.print_freq = N_epoch
        
        if plot_freq != None:
            self.plot_freq = plot_freq
        else:
            self.plot_freq = 1
        
        if use_pretrained == True:
            self.blob = blob
        
        self.exphistory  = {
            'x': [],
            'logq': [],
            'logp': [],
            'accepted': [] 
        }

    
    def train_step(self):
        layers, prior = self.model['layers'], self.model['prior']                                     
        self.optimizer.zero_grad()                                                  

        x, logq = apply_flow_to_prior(prior, layers, batch_size = self.batch_size)  
        logp = - self.action(x)                                                     
        loss = self.criterion(logp, logq)                                             
        loss.backward()                                                        

        self.optimizer.step()                                                       

        self.history['loss'].append(grab(loss))                                     
        self.history['logp'].append(grab(logp))                                     
        self.history['logq'].append(grab(logq))
        self.history['ess'].append(grab(compute_ess(logp, logq)))
    
    def print_metrics(self, era, epoch, avg_last_N_epochs):                  
        print(f' == Era {era} | Epoch {epoch} metrics')                      
        for key, val in self.history.items():
            avgd = np.mean(val[-avg_last_N_epochs:])
            print(f'\t{key} {avgd:g}')
    
    def train(self):
        [plt.close(plt.figure(fignum)) for fignum in plt.get_fignums()]     #Closing all existing figure
        live_plot = init_live_plot(self.N_epoch, self.N_era)                #Producing the parameter for live plot

        for era in range(self.N_era):
            for epoch in range(self.N_epoch):
                self.train_step()                                           #Perform one training epoch

                if epoch % self.print_freq == 0:                            #Print the history once a time
                    self.print_metrics(era, epoch, avg_last_N_epochs = self.print_freq)
                
                if epoch % self.plot_freq == 0:                             #Update the plot once a time
                    update_plots(self.history, **live_plot)
    
    def run(self, train = True):
        if self.use_pretrained == True:
            print('Skipping training')
            print('Loading pre-trained model')
            phi4_trained_weights = torch.load(io.BytesIO(base64.b64decode(self.blob.strip())), 
                                              map_location = torch.device('cpu'))
            self.model['layers'].load_state_dict(phi4_trained_weights)
            if torch_device == 'cuda':
                self.model['layers'].cuda()
        if train == True:
            self.train()
        
        serialized_model = io.BytesIO()
        torch.save(self.model['layers'].state_dict(), serialized_model)
        self.blob = base64.b64encode(serialized_model.getbuffer()).decode('utf-8')

    def serial_sample_generator(self, N_samples):
        layers, prior = self.model['layers'], self.model['prior']                                     
        layers.eval()
        x, logq, logp = None, None, None
        for i in tqdm(range(N_samples)):
            batch_i = i % self.batch_size
            if batch_i == 0:
                x, logq = apply_flow_to_prior(prior, layers, batch_size = self.batch_size)
                logp = - self.action(x)
            yield x[batch_i], logq[batch_i], logp[batch_i]
    
    def make_mcmc_ensemble(self, N_samples):
        history  = {
            'x': [],
            'logq': [],
            'logp': [],
            'accepted': [] 
        }

        sample_gen = self.serial_sample_generator(N_samples)
        for new_x, new_logq, new_logp in sample_gen:
            new_x = grab(new_x)                                 # Detach all tensor here to save memory
            new_logq = grab(new_logq)                           # Based on my experience, not performing detach
            new_logp = grab(new_logp)                           # may crash the computer when we use large number
            if len(history['logp']) == 0:                       # of samples, due to the computation graph that
                accepted = True                                 # follows the torch.tensor object
            else:
                last_logp = history['logp'][-1]
                last_logq = history['logq'][-1]
                p_accept = np.exp((new_logp - new_logq) - (last_logp - last_logq))   #Default probability to accept
                p_accept = min(1, p_accept)                     #If p > 1 (Delta < 0), just set p = 1
                draw = np.random.uniform(0, 1)                  #generate random number between 0 and 1
                if draw < p_accept:                             #If Delta < 0, always accept
                    accepted = True                             #If not, tend to accept as long as Delta << 1
                else:
                    accepted = False
                    new_x = history['x'][-1]
                    new_logp = last_logp
                    new_logq = last_logq
            history['logp'].append(new_logp)
            history['logq'].append(new_logq)
            history['x'].append(new_x)
            history['accepted'].append(accepted)
        self.mcmc_history = history
        return history   
    






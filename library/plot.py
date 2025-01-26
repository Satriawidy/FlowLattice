from .initial import *
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

def init_live_plot(N_epoch, N_era, dpi = 125, figsize = (8, 4)):    #Function to display live result of the training
    fig, ax_ess = plt.subplots(1, 1, dpi = dpi, figsize = figsize)   #Create figure and axes for the live plot
    plt.xlim(0, N_era * N_epoch)                                    #Adapt the xlim depending on num of epochs
    plt.ylim(0, 1)

    ess_line = plt.plot([0], [0], alpha = 0.5)                      #Create initial line for validation metric
    plt.grid(False)
    plt.ylabel('ESS')

    ax_loss = ax_ess.twinx()
    loss_line = plt.plot([0], [0], alpha = 0.5, c = 'orange')       #Create initial line for loss value
    plt.grid(False)
    plt.ylabel('Loss')

    plt.xlabel('Epoch')                                             #The x-axis is num of epoch

    display_id = display(fig, display_id = True)                    #For displaying the figure in IPython
    return dict(                                                    #Return dict containing information to plot
        fig = fig, ax_ess = ax_ess, ax_loss = ax_loss,
        ess_line = ess_line, loss_line = loss_line,
        display_id = display_id
    )

def moving_average(x, window = 10):                                 #Function to compute average value of some variables
    if len(x) < window:                                             #over some number of windows (last iteration)
        return np.mean(x, keepdims = True)                          #If num of iteration < window, just return average
    else:
        return np.convolve(x, np.ones(window), 'valid') / window    #If num of iteration > window, return average over
                                                                    #last few iterations

def update_plots(history, fig, ax_ess, ax_loss, ess_line, loss_line, display_id):   #Real moving plot
    Y = np.array(history['ess'])                                    #Plotting the validation metric
    Y = moving_average(Y, window = 15)                              #Computing the moving average
    ess_line[0].set_ydata(Y)                                        #Putting the moving average on the line
    ess_line[0].set_xdata(np.arange(len(Y)))

    Y = history['loss']                                             #Now for the loss value
    Y = moving_average(Y, window = 15)
    loss_line[0].set_ydata(np.array(Y))
    loss_line[0].set_xdata(np.arange(len(Y)))
    ax_loss.relim()                                                 #Change the y-limit for loss as it will update wildly
    ax_loss.autoscale_view()                                        #Autoscale the y-limit for loss for the same reason
    fig.canvas.draw()                                               #Redraw the frame
    display_id.update(fig)                                          #Update the plot

def sample_plot(z, shape = (4, 4), dpi = 125, figsize = (4,4)):
    fig, ax = plt.subplots(*shape, dpi = dpi, figsize = figsize)
    for i in range(shape[0]):
        for j in range(shape[1]):
            ind = i * shape[1] + j
            ax[i, j].imshow(np.tanh(z[ind]), vmin = -1, vmax = 1, cmap = 'viridis')
            ax[i, j].axes.xaxis.set_visible(False)
            ax[i, j].axes.yaxis.set_visible(False)
    plt.show()

def correlation_plot(S_eff, S):
    fit_b = np.mean(S) - np.mean(S_eff)

    print(f'slope 1 linear regression S = S_eff + {fit_b:.4f}')

    a = min(S_eff)
    b = max(S_eff)

    fig, ax = plt.subplots(1, 1, dpi = 125, figsize = (4,4))
    ax.hist2d(S_eff, S, bins = 20, range = [[a, b], [a + fit_b, b + fit_b]])
    xs = np.linspace(a, b, num = 4, endpoint = True)
    ax.plot(xs, xs + fit_b, ':', color = 'w', label = 'slope 1 fit')
    ax.set_xlabel(r'$S_{\mathrm{eff}} = -\log~q(x)$')
    ax.set_ylabel(r'$S(x)$')
    ax.set_aspect('equal')
    plt.legend(prop={'size': 6})
    plt.show()

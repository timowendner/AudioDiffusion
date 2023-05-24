#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def plot_variance_scheduler():
    
    T = 250
    t = np.linspace(0, 1, T)
    n = np.linspace(0, (250+1), T)
    
    beta_start = 0.001
    beta_end = 0.04
    # beta_end = 0.02
    
    beta_linear = beta_start + (beta_end - beta_start) * t
    alpha_linear = 1 - beta_linear
    alpha_linear_hat = np.cumprod(alpha_linear, 0)
    
    beta_quadratic = beta_start + (beta_end - beta_start) * t**2
    alpha_quadratic = 1 - beta_quadratic
    alpha_quadratic_hat = np.cumprod(alpha_quadratic, 0)
    
    a = 0.15
    beta_sigmoidal = beta_start + (beta_end - beta_start) * 1 / \
            (1 + np.e ** -(a/(beta_end - beta_start)*(t - beta_start-0.5)))
    alpha_sigmoidal = 1 - beta_sigmoidal
    alpha_sigmoidal_hat = np.cumprod(alpha_sigmoidal, 0)
    
    t = np.linspace(0, (250+1), T)
    plt.plot(t, alpha_linear_hat, label='linear')
    plt.plot(t, alpha_quadratic_hat, label='quadratic')
    plt.plot(t, alpha_sigmoidal_hat, label='sigmoidal')
    plt.legend()
    plt.xlabel('diffusion steps t')
    # plt.ylabel('$\alpha$')
    plt.ylabel(r"$\bar{\alpha_t}$")
    # plt.title('')
    plt.savefig(os.path.join('plots', 'scheduler.png'))
    
    return None

def plot_FAD_augmentation_comparison():
    
    read_csv = os.path.join('plots', 'FAD_augmented_orig.csv')
    save_csv = os.path.join('plots', 'FAD_augmented_cleaned.csv')

    results = pd.read_csv(read_csv)
    results = results.iloc[1:24, 1:] # select rows were we have enough values for all columns
    results = results.iloc[:,[1,5,9]] # select relevant columns
    results.rename(columns={"Unnamed: 2": "No augmentation", "Unnamed: 6": "Augmentation 1", "Unnamed: 10": "Augmentation 2"}, inplace=True)
    results.to_csv(save_csv, index=None)
    
    # internal pd bug when creating boxplot directly from results
    df = pd.read_csv(save_csv)
    boxplot = df.boxplot(column=['No augmentation', 'Augmentation 1', 'Augmentation 2'])  
    boxplot.figure.savefig(os.path.join('plots', 'augmentation_comparison.png'))
    
    return None    


# plot_variance_scheduler()
# plot_FAD_augmentation_comparison()
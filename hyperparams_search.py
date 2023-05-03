import os
import shutil
import tqdm

import torch
from torch.utils.data import DataLoader
import numpy as np
from scipy.io.wavfile import write
import optuna

from hp_dataloader import AudioDataset
from hp_diffusion import Diffusion
from hp_model import UNet 
from hp_utils import save_model, save_samples
# from fad_score import FADWrapper
from eval import FADWrapper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
counter = 0

def update_config_savesamples(config):
    
    global counter
    config.output_path = '{}/generated_files'.format(config.model_name)
    config.save_samples = '{}/{}'.format(config.model_name, counter)
    config.create_label = [0,1,2,3,4,5,6]
    config.create_count = 100
    config.samples_to_keep = 5
    counter += 1

def fad(model, hp_config, diffusion, labels = [0,1,2,3,4,5,6]):
    config = update_config_savesamples(hp_config)
    if os.path.exists(config.output_path):
        shutil.rmtree(config.output_path)
    os.makedirs(config.output_path)
    for label in labels:
        config.create_label = label
        for i in range(10):
            torch.cuda.empty_cache()
            save_samples(model, diffusion, config, it=i)
    fad_wrapper = FADWrapper.FADWrapper(generated_audio_samples_dir=config.output_path, ground_truth_audio_samples_dir="fad_score/data/eval")
    fd = fad_wrapper.compute_fad()
    print(fd)
    
    config.output_path = config.save_samples
    config.create_count = config.samples_to_keep
    save_samples(model, diffusion, config)

    return np.mean(list(fd['FAD']))

def objective(trial):
    # Define the hyperparameters to search over
    hp_config = {
        # hyperparams
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
        'lr': trial.suggest_float('lr', 1e-5, 1e-3),
        # diffusion
        'step_count': trial.suggest_int('step_count', 250, 500, 50),
        'beta_start': trial.suggest_float('beta_start', 1e-5, 1e-3),
        'beta_end': trial.suggest_float('beta_end', 0.01, 0.04),
        'beta_schedule': trial.suggest_categorical('beta_schedule', ['quadratic', 'linear', 'sigmoid']),
        'beta_sigmoid': trial.suggest_float('beta_sigmoid', 0.15, 0.30),
        # sampling
        'create_loop': trial.suggest_int('create_loop', 1, 3),
    }
    
    # data loading and diffusion
    diffusion = Diffusion(hp_config, device)
    dataset = AudioDataset(diffusion, device)
    train_loader = DataLoader(dataset, batch_size=hp_config['batch_size'],
                              shuffle=True, num_workers=2)
    
    # model and optimizer
    model = UNet(device, hp_config)
    model.to(device)
    optimizer = None

    if(hp_config['optimizer'] == "Adam"):
        optimizer = torch.optim.Adam(model.parameters(), lr=hp_config['lr'])
    elif(hp_config['optimizer'] == "RMSprop"):
        optimizer = torch.optim.RMSprop(model.parameters(), lr=hp_config['lr'])
    elif(hp_config['optimizer'] == "SGD"):
        optimizer = torch.optim.SGD(model.parameters(), lr=hp_config['lr'])

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'num_parameters: {num_params:,}')
    
    # training
    model.train()
    num_epochs = 1 # 250
    num_batches = len(train_loader)
    criterion = torch.nn.MSELoss()
    for epoch in tqdm.tqdm(range(num_epochs)):
        #  # print the epoch and current time
        # time_now = datetime.datetime.now().strftime("%H:%M")
        # print(f"Start Epoch: {epoch + 1}/{num_epochs}   {time_now}")
        
        # loop through the training loader
        for i, (samples, targets, timesteps, labels) in enumerate(train_loader):
            # forward
            outputs = model(samples, timesteps, labels)
            loss = criterion(outputs, targets)

            # calculate gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if not i % 50:
                print(f'Epoch [{epoch + 1}/{num_epochs}]',
                      f'Batch [{i + 1}/{num_batches}]',
                      f'Loss: {loss.item():.4f}')
    
    mean_FAD = fad(model, hp_config, diffusion)
            
    return mean_FAD


if __name__ == '__main__':
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=1) # TODO change
    trial = study.best_trial
    print('FAD: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))
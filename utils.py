import torch
import os
import datetime
import pickle as pkl
import numpy as np
from scipy.io.wavfile import write

from os.path import join, isfile, getmtime, exists
from model import UNet


def save_model(model, optimizer, config):
    if not exists(config.model_path):
        os.makedirs(config.model_path)

    # get the time now
    time_now = datetime.datetime.now()
    time_now = time_now.strftime("%d%b_%H%M")

    # save the model
    filepath = join(config.model_path, f"{config.model_name}_{time_now}.p")

    # define the config arguments to be saved
    change_config = ("audio_length", "beta_start", "beta_end", "beta_schedule", "step_count",
                     "model_layers", "model_out", "model_kernel", "model_scale", 'current_epoch')
    change_config = {arg: getattr(config, arg) for arg in change_config}

    # save everything
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': change_config,
    }, filepath)


def save_samples(model, diffusion, config):
    if not exists(config.output_path):
        os.makedirs(config.output_path)

    # get the time now
    time_now = datetime.datetime.now()
    time_now = time_now.strftime("%d%b_%H%M")

    foldernames = {
        0: 'dog_bark',
        1: 'footstep',
        2: 'gunshot',
        3: 'keyboard',
        4: 'moving_motor_vehicle',
        5: 'rain',
        6: 'sneeze_cough',
    }

    # remove the current wav
    folderpath = join(config.output_path, foldernames[config.create_label])
    if not exists(folderpath):
        os.makedirs(folderpath)
    for f in os.listdir(folderpath):
        os.remove(join(folderpath, f))

    # create a new datapoint
    for i in range(config.create_count):
        print(f'Start creating output: {i+1}')
        # create a new waveform
        waveform = diffusion.sample(model, config)
        waveform = waveform[0, 0].to('cpu').numpy()

        # remove any clicks at the start and end
        cut = 200
        lin = np.cos(np.linspace(1, 0, cut)**2 * np.pi) / 2 + 0.5
        waveform[:cut] = waveform[:cut] * lin
        waveform[-cut:] = waveform[-cut:] * lin[::-1]

        # normalize the waveform and write the file
        waveform = waveform / np.max(waveform) * 0.65
        scaled = np.int16(waveform * 32767)
        name = f'output_{time_now}_{i:2}.wav'
        write(join(folderpath, name), 22050, scaled)


def create_model(config, load=False, lr=False):
    if not exists(config.model_path):
        os.makedirs(config.model_path)

    files = [join(config.model_path, f) for f in os.listdir(
        config.model_path) if isfile(join(config.model_path, f))]
    files = sorted(files, key=getmtime)
    model = UNet(config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    if not load or len(files) == 0:
        config.current_epoch = 0
        return model, config, optimizer

    filepath = files[-1]
    print(f'Load model: {filepath}')
    loaded = torch.load(filepath)

    # copy the config file
    for argument, value in loaded['config'].items():
        setattr(config, argument, value)

    model = UNet(config).to(config.device)
    model.load_state_dict(loaded['model'])
    if lr:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer.load_state_dict(loaded['optimizer'])

    return model, config, optimizer

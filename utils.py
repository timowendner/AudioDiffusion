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

    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': config.current_epoch,
    }, filepath)


def save_model(model, optimizer, config):
    if not exists(config.model_path):
        os.makedirs(config.model_path)

    # get the time now
    time_now = datetime.datetime.now()
    time_now = time_now.strftime("%d%b_%H%M")

    # save the model
    filepath = join(config.model_path, f"{config.model_name}_{time_now}.p")

    # define the config arguments to be saved
    change_config = ("audio_length", "data_targetSD", "model_layers",
                     "model_out", "model_kernel", "model_scale", 'current_epoch')
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

    # create a new datapoint
    output = diffusion.sample(model, config)
    output = output.to('cpu')

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

    for i, data in enumerate(output):
        data = data[0, :].numpy()
        data = data / np.max(data) * 0.9
        scaled = np.int16(data * 32767)
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


def load_model(model, optimizer, config):
    if not exists(config.model_path):
        os.makedirs(config.model_path)
        return None

    files = [join(config.model_path, f) for f in os.listdir(
        config.model_path) if isfile(join(config.model_path, f))]
    files = sorted(files, key=getmtime)

    if len(files) == 0:
        return None
    filepath = files[-1]

    print(f'Load model: {filepath}')

    loaded = torch.load(filepath)
    model.load_state_dict(loaded['model'])
    if not config.change_lr:
        optimizer.load_state_dict(loaded['optimizer'])
    config.current_epoch = loaded['epoch']

    # with open(filepath, 'rb') as f:
    #     model = pkl.load(f)
    return model, optimizer

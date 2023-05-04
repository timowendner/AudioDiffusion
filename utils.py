import torch
import os
import datetime
import pickle as pkl
import numpy as np
from scipy.io.wavfile import write

from os.path import join, isfile, getmtime, exists


def save_model(model, optimizer, config, epoch):
    if not exists(config.model_path):
        os.makedirs(config.model_path)

    # get the time now
    time_now = datetime.datetime.now()
    time_now = time_now.strftime("%d%b_%H%M")

    # save the model
    filepath = join(config.model_path, f"{config.model_name}_{epoch}eps.p")

    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': config.current_epoch,
    }, filepath)


def save_samples(model, diffusion, config, it=0):
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
    #for f in os.listdir(folderpath):
     #   os.remove(join(folderpath, f))

    for i, data in enumerate(output):
        i = i + 10*it
        data = data[0, :].numpy()
        data = data / np.max(data) * 0.9
        scaled = np.int16(data * 32767)
        name = f'output_{time_now}_{i:2}.wav'
        write(join(folderpath, name), 22050, scaled)


def load_model(model, optimizer, config): # TODO fix loading
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

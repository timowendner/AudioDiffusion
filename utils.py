import torch
import os
import datetime
import pickle as pkl
import numpy as np
from scipy.io.wavfile import write

from os.path import join, isfile, getmtime, exists


def save_model(model, config):
    if not exists(config.model):
        os.makedirs(config.model)

    # get the time now
    time_now = datetime.datetime.now()
    time_now = time_now.strftime("%d%b_%H%M")

    # save the model
    filepath = join(config.model, f"{model.name}_{time_now}.p")
    with open(filepath, 'wb') as f:
        pkl.dump(model, f)


def save_samples(diffusion, config, label, count, loop=1):
    if not exists(config.output):
        os.makedirs(config.output)
    # create a new datapoint
    output = diffusion.sample([label] * count, loop=loop)
    output = output.to('cpu')

    # get the time now
    time_now = datetime.datetime.now()
    time_now = time_now.strftime("%d%b_%H%M")

    # foldernames
    foldernames = {
        1: 'dog_bark',
        2: 'sneeze_cough',
        3: 'rain',
        4: 'moving_motor_vehicle',
        5: 'keyboard',
        6: 'gunshot',
        7: 'footstep',
    }

    # remove the current wav
    folderpath = join(config.output, foldernames[label])
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


def load_model(config):
    if not exists(config.model):
        os.makedirs(config.model)
        return None

    files = [join(config.model, f) for f in os.listdir(
        config.model) if isfile(join(config.model, f))]
    files = sorted(files, key=getmtime)

    if len(files) == 0:
        return None

    with open(files[-1], 'rb') as f:
        model = pkl.load(f)
    return model


class Config:
    def __init__(self, model_path, data_path, output_path, label_path, label_train, step_count, label_count, lr) -> None:
        self.model = model_path
        self.output = output_path
        self.data = data_path
        self.labels = label_path
        self.label_train = label_train
        self.step_count = step_count
        self.label_count = label_count
        self.lr = lr

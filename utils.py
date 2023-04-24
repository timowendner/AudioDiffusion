import torch
import os
import datetime
import pickle as pkl
import numpy as np
from scipy.io.wavfile import write

from os.path import join, isfile, getmtime


def save_model(model, path):
    if not os.path.exists(path.model):
        os.makedirs(path.model)

    # get the time now
    time_now = datetime.datetime.now()
    time_now = time_now.strftime("%d%b_%H%M")

    # save the model
    filepath = join(path.model, f"{model.name}_{time_now}.p")
    torch.save(model.state_dict(), filepath)


def save_samples(diffusion, path, label, count, loop=1):
    if not os.path.exists(path.output):
        os.makedirs(path.output)
    # create a new datapoint
    output = diffusion.sample([label] * count, loop=loop)
    output.to('cpu')

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
    folderpath = join(path.output, foldernames[label])
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
    for f in os.listdir(folderpath):
        os.remove(join(folderpath, f))

    for i, data in enumerate(output):
        data = data[0, :].numpy()
        data = data / np.max(data) * 0.9
        scaled = np.int16(data * 32767)
        name = f'{output[:-4]}_{time_now}_{i:2}.wav'
        write(join(folderpath, name), 22050, scaled)


def load_model(empty_model, path, specific_model=None):
    if not os.path.exists(path.model):
        os.makedirs(path.model)
        return None

    if specific_model is not None:
        empty_model.load_state_dict(torch.load(
            join(path.model, specific_model), map_location=empty_model.device))

    files = []
    for f in os.listdir(path.model):
        if isfile(join(path.model, f)) and f[:len(empty_model.name)] == empty_model.name:
            files.append(join(path.model, f))

    if len(files) == 0:
        return None

    # sort files based on modification time
    files = sorted(files, key=lambda f: getmtime(f))

    empty_model.load_state_dict(torch.load(
        files[-1], map_location=empty_model.device))


class Path:
    def __init__(self, model_path, data_path, output_path, label_path) -> None:
        self.model = model_path
        self.output = output_path
        self.data = data_path
        self.labels = label_path

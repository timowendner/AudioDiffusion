import torch
import os
import datetime
import pickle as pkl

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


def save_samples(diffusion, path, labels: list):
    if not os.path.exists(path.output):
        os.makedirs(path.output)
    # create a new datapoint
    x = diffusion.sample(labels)
    x.to('cpu')

    # get the time now
    time_now = datetime.datetime.now()
    time_now = time_now.strftime("%d%b_%H%M")

    # create the filepath
    filepath = join(path.output, f'output_{diffusion.model.name}_{time_now}')

    # save the data to a pickle file
    with open(time_now, 'wb') as f:
        pkl.dump(x, f)


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
    def __init__(self, model_path, data_path, output_path) -> None:
        self.model = model_path
        self.output = output_path
        self.data = data_path

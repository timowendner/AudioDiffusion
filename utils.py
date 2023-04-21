import torch
import os
import datetime
import pickle as pkl

from os.path import join, isfile, getmtime


def save_model(model, modelpath):
    if not os.path.exists(modelpath):
        os.makedirs(modelpath)

    # get the time now
    time_now = datetime.datetime.now()
    time_now = time_now.strftime("%d%b_%H%M")

    # save the model
    filepath = join(modelpath, f"{model.name}_{time_now}.p")
    torch.save(model.state_dict(), filepath)


def save_samples(diffusion, outputpath: str, labels: list):
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    # create a new datapoint
    x = diffusion.sample(labels)
    x.to('cpu')

    # get the time now
    time_now = datetime.datetime.now()
    time_now = time_now.strftime("%d%b_%H%M")

    # create the filepath
    filepath = join(outputpath, f'output_{diffusion.model.name}_{time_now}')

    # save the data to a pickle file
    with open(time_now, 'wb') as f:
        pkl.dump(x, f)


def load_model(empty_model, modelpath, specific_model=None):
    if not os.path.exists(modelpath):
        os.makedirs(modelpath)
        return None

    if specific_model is not None:
        empty_model.load_state_dict(torch.load(
            join(modelpath, specific_model), map_location=empty_model.device))

    files = [f for f in os.listdir(modelpath) if isfile(join(modelpath, f))]
    files = [f for f in files if f[:len(empty_model)] == empty_model]

    if len(files) == 0:
        return None

    # sort files based on modification time
    files = sorted(files, key=lambda f: getmtime(join(modelpath, f)))

    empty_model.load_state_dict(torch.load(
        join(modelpath, files[-1]), map_location=empty_model.device))

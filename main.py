import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pickle as pkl
import json

import argparse
import datetime
import time
import os

from dataloader import AudioDataset
from diffusion import Diffusion
from utils import save_model, load_model, save_samples


def train_network(model, optimizer, diffusion, config):
    # create the dataset
    dataset = AudioDataset(diffusion, config, model.device)
    train_loader = DataLoader(dataset, batch_size=16,
                              shuffle=True, num_workers=0)
    # Train the model
    model.train()
    total_step = len(train_loader)
    mse = torch.nn.MSELoss()

    start_time = time.time()
    for epoch in range(config.num_epochs):
        # print the epoch and current time
        time_now = datetime.datetime.now()
        time_now = time_now.strftime("%H:%M")
        print(f"Start Epoch: {epoch + 1}/{config.num_epochs}   {time_now}")

        # loop through the training loader
        for i, (model_input, targets, t, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(model_input, t, labels)
            loss = mse(outputs, targets)

            # calculate gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
                print(f'Epoch [{epoch + 1}/{config.num_epochs}]',
                      f'Step [{i + 1}/{total_step}]',
                      f'Loss: {loss.item():.4f}')

        # add the number of epochs
        config.current_epoch += 1

        # save the model if enough time has passed
        if abs(time.time() - start_time) >= 5*60 or epoch == config.num_epochs - 1:
            save_model(model, optimizer, config)
            start_time = time.time()
    return model, optimizer


def main():
    # Load JSON config file
    with open(args.config_path) as f:
        config_json = json.load(f)

    # create the config file
    class Config:
        pass
    config = Config()
    for key, data in config_json.items():
        setattr(config, key, data)

    # set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device

    # load the right model
    if config.model_number == 1:
        from model import UNet
    elif config.model_number == 2:
        from model2 import UNet
    elif config.model_number == 3:
        from model3 import UNet

    # create the model and the diffusion
    model = UNet(device, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    diffusion = Diffusion(config)
    config.current_epoch = 0

    # load the latest model
    if args.load:
        model, optimizer = load_model(model, optimizer, config)

    # print the number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Number of trainable parameters: {num_params:,}, with epoch {config.current_epoch}")

    # train the network
    if args.train:
        model, optimizer = train_network(model, optimizer, diffusion, config)

    # create new samples
    save_samples(model, diffusion, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diffusion Model')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--config_path', type=str,
                        help='Path to the configuration file')
    parser.add_argument('--load', action='store_true',
                        help='load a model')
    args = parser.parse_args()

    main()

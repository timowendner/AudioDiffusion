import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pickle as pkl

import argparse
import datetime
import time
import os

from model import UNet
from dataloader import AudioDataset
from diffusion import Diffusion
from utils import save_model, load_model, save_samples, Config


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
    # load the files
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config(
        data_path='/content/drive/MyDrive/AudioDiffusion/data',
        model_path='/content/drive/MyDrive/AudioDiffusion/models',
        output_path='/content/drive/MyDrive/AudioDiffusion/output',
        label_path={
            0: 'DogBark',
            1: 'Footstep',
            2: 'GunShot',
            3: 'Keyboard',
            4: 'MovingMotorVehicle',
            5: 'Rain',
            6: 'Sneeze_Cough',
        },  # all label folders
        label_train={0, },  # what labels to train for
        label_count=7,  # how many samples are there
        step_count=100,  # how many diffusion steps do we have
        lr=0.001,  # learning rate
        create_loop=args.loop,  # repeat every timestamp while creating samples
        create_label=args.label,  # what sample to create
        create_count=args.count,  # how many samples to create
        num_epochs=1000,  # epochs to train
        audio_length=88200,  # audio length
        beta_start=1e-4,  # diffusion start
        beta_end=0.02,  # diffusion end
        beta_sigmoid=0.15,  # diffusion sigmoid
    )
    config.model_name = 'aperture'
    config.device = device

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
    parser.add_argument('--load', action='store_true',
                        help='load a model')
    parser.add_argument('--model', type=int, default=1,
                        help='choose the model')
    parser.add_argument('--label', type=int, default=1, help='Label to sample')
    parser.add_argument('--count', type=int, default=1,
                        help='How many samples to create')
    parser.add_argument('--loop', type=int, default=1,
                        help='How often the diffusion should happen')
    args = parser.parse_args()

    if args.model == 2:
        from model2 import UNet
    elif args.model == 3:
        from model3 import UNet
    main()

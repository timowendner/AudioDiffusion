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
from utils import save_model, load_model, save_samples


def train_network(model, file_path, diffusion, num_epochs):
    # create the dataset
    dataset = AudioDataset(file_path, model.device,  diffusion)
    train_loader = DataLoader(dataset, batch_size=16,
                              shuffle=True, num_workers=0)
    # Train the model
    model.train()
    total_step = len(train_loader)
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()
    for epoch in range(num_epochs):
        # print the epoch and current time
        time_now = datetime.datetime.now()
        time_now = time_now.strftime("%H:%M")
        print(f"Start Epoch: {epoch + 1}/{num_epochs}   {time_now}")

        # loop through the training loader
        for i, (model_input, targets, t, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(model_input, t, labels)
            loss = mse(outputs, targets)

            # calculate gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # nn.utils.clip_grad_norm_(model.parameters(), 0.001)

            if (i + 1) % 50 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}]',
                      f'Step [{i + 1}/{total_step}]',
                      f'Loss: {loss.item():.4f}')

        # add the number of epochs
        model.epoch += 1

        # save the model if enough time has passed
        if abs(time.time() - start_time) >= 5*60 or epoch == num_epochs - 1:
            save_model(model)
            start_time = time.time()

        # test the model and plot a example image
        # test_network(model, loaders["test"], device)
        # plot(1, loaders["train"], model, device)


def main():
    # load the files
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = '/content/drive/MyDrive/AudioDiffusion/data'
    model_path = '/content/drive/MyDrive/AudioDiffusion/models'
    output_path = '/content/drive/MyDrive/AudioDiffusion/output'
    # data_path = '/Users/timowendner/Programming/AudioDiffusion/Data/DogBark'

    # create the model and the diffusion
    steps = 100
    labels = 7
    model = UNet(device, steps, labels).to(device)
    model.name = 'duplex'
    model.epoch = 0
    diffusion = Diffusion(model, length=88200, steps=steps)

    # print the number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    # load a model
    load_model(model, model_path)

    # train the network
    if args.train:
        train_network(model, data_path, diffusion, num_epochs=1000)

    # create new samples
    labels = [1, 1, 2, 3, 4, 5, 6, 7]
    save_samples(diffusion, output_path, labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diffusion Model')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--model2', action='store_true',
                        help='Use model2 instead of model1')
    args = parser.parse_args()

    if args.model2:
        from model2 import UNet
    main()

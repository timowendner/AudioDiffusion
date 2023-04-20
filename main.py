import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pickle as pkl

from dataloader import AudioDataset
from model import UNet
import datetime
import time
import os
from diffusion import Diffusion


def save_model(model):
    # save the model with a timestamp
    if not os.path.exists('/content/drive/MyDrive/AudioDiffusion/models'):
        os.makedirs('/content/drive/MyDrive/AudioDiffusion/models')

    # get the time now
    time_now = datetime.datetime.now()
    time_now = time_now.strftime("%d_%b_%H%M")

    # save the model
    filepath = f"/content/drive/MyDrive/AudioDiffusion/models/{model.name}_{time_now}.p"
    torch.save(model.state_dict(), filepath)


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

    for epoch in range(num_epochs):
        # print the epoch and current time
        time_now = datetime.datetime.now()
        start_time = time.time()
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

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}]',
                      f'Step [{i + 1}/{total_step}]',
                      f'Loss: {loss.item():.4f}')

        # save the model if enough time has passed
        if time.time() - start_time >= 5*60 or epoch == num_epochs - 1:
            save_model(model)
            start_time = time.time()

        # test the model and plot a example image
        # test_network(model, loaders["test"], device)
        # plot(1, loaders["train"], model, device)


def sample(diffusion, outputpath: str, labels: list):
    # create a new datapoint
    x = diffusion.sample(labels)
    x.to('cpu')

    # save the data to a pickle file
    with open(outputpath, 'wb') as f:
        pkl.dump(x, f)


def main():
    # load the files
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    file_path = '/content/drive/MyDrive/AudioDiffusion/data'
    # file_path = '/Users/timowendner/Programming/AudioDiffusion/Data/DogBark'

    # create the model and the diffusion
    steps = 1000
    labels = 7
    model = UNet(device, steps, labels).to(device)
    model.name = 'duplex'
    diffusion = Diffusion(model, length=88200)

    # # load a model
    # modelpath = '/content/drive/MyDrive/AudioDiffusion/models/glados_20_Apr_0044.p'
    # model.load_state_dict(torch.load(modelpath, map_location=device))

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    # train the network
    train_network(model, file_path, diffusion, num_epochs=1000)

    # # create new samples
    # outputpath = '/content/drive/MyDrive/AudioDiffusion/output/output2.pkl'
    # labels = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4,
    #           4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7]
    # sample(diffusion, outputpath, labels)

    # # load the data from the pickle file
    # with open('data.pkl', 'rb') as f:
    #     loaded_data = pkl.load(f)


if __name__ == '__main__':
    main()

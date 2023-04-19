import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pickle as pkl

from dataloader import AudioDataset
from model import UNet
import datetime
import os
from diffusion import Diffusion


def save_model(model):
    # save the model with a timestamp
    if not os.path.exists('trained_models'):
        os.makedirs('trained_models')

    # get the time now
    time_now = datetime.datetime.now()
    time_now = time_now.strftime("%d_%b_%H%M")

    # save the model
    filepath = f"/content/drive/MyDrive/AudioDiffusionModels/testchamber_{time_now}.p"
    torch.save(model.state_dict(), filepath)


def train_network(model, train_loader, num_epochs):
    # Train the model
    model.train()
    total_step = len(train_loader)
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        # print the epoch and current time
        time_now = datetime.datetime.now()
        time_now = time_now.strftime("%H:%M")
        print(f"Start Epoch: {epoch + 1}/{num_epochs}   {time_now}")

        # loop through the training loader
        for i, (model_input, targets, t) in enumerate(train_loader):
            # Forward pass
            outputs = model(model_input, t, 1)
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
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            save_model(model)

        # test the model and plot a example image
        # test_network(model, loaders["test"], device)
        # plot(1, loaders["train"], model, device)


def main():
    # load the files
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    file_path = '/content/drive/MyDrive/Data/DogBark'
    # file_path = '/Users/timowendner/Programming/AudioDiffusion/Data/DogBark'

    # create the model and the dataloader
    model = UNet(device).to(device)
    diffusion = Diffusion(model, length=88200)
    dataset = AudioDataset(file_path, device,  diffusion)
    train_loader = DataLoader(dataset, batch_size=16,
                              shuffle=True, num_workers=0)
    # # Play the first audio
    # import sounddevice as sd
    # audiofile = dataset[0][0].numpy()[0, :]
    # print(audiofile.shape)
    # sd.play(audiofile, samplerate=22050)
    # sd.wait()

    num_epochs = 1000

    # train the network
    # train_network(model, train_loader, num_epochs)

    # create new samples
    outputpath = '/content/drive/MyDrive/AudioDiffusion/output/output1.pkl'
    modelpath = '/content/drive/MyDrive/AudioDiffusionModels/testchamber_18_Apr_2229.p'

    model.load_state_dict(torch.load(modelpath, map_location=device))

    # create a new datapoint
    x = diffusion.sample(100, 1)
    # save the data to a pickle file
    with open(outputpath, 'wb') as f:
        pkl.dump(x, f)

    # # load the data from the pickle file
    # with open('data.pkl', 'rb') as f:
    #     loaded_data = pkl.load(f)


if __name__ == '__main__':
    main()

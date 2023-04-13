import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from dataloader import AudioDataset
from model import UNet
import datetime
import os


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


def train_network(model, train_loader, num_epochs, optimizer, loss_func):
    # Train the model
    model.train()
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        print(f"Start Epoch: {epoch + 1}/{num_epochs}")
        for i, (model_input, targets) in enumerate(train_loader):
            # Forward pass
            outputs = model(model_input)
            loss = loss_func(outputs, targets)

            # calculate gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # nn.utils.clip_grad_norm_(model.parameters(), 0.001)

            if (i + 1) % 10 == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')
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
    dataset = AudioDataset(file_path, device)

    # # Play the first audio
    # import sounddevice as sd
    # audiofile = dataset[0][0].numpy()[0, :]
    # print(audiofile.shape)
    # sd.play(audiofile, samplerate=22050)
    # sd.wait()

    # create the dataloader
    train_loader = DataLoader(dataset, batch_size=16,
                              shuffle=True, num_workers=0)

    # create the model
    model = UNet().to(device)

    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,)
    num_epochs = 100

    # train the network
    train_network(model, train_loader, num_epochs, optimizer, loss_func)


if __name__ == '__main__':
    main()

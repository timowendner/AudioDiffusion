import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from dataloader import AudioDataset
from model import UNet


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
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            if not torch.isnan(loss):
                loss.backward()
                optimizer.step()

            if (i + 1) % 1 == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')
        # if epoch % 2 == 0:
        #     save_model(model)

        # test the model and plot a example image
        # test_network(model, loaders["test"], device)
        # plot(1, loaders["train"], model, device)


def main():
    # load the files
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    file_path = '/content/drive/MyDrive/Data/DogBark'
    dataset = AudioDataset(file_path, device)

    # create the dataloaders
    train_loader = DataLoader(dataset, batch_size=16,
                              shuffle=True, num_workers=0)

    # create the model
    model = UNet().to(device)
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001,)
    num_epochs = 100

    # train the network
    train_network(model, train_loader, num_epochs, optimizer, loss_func)

    # # Define loss function and optimizer
    # criterion = torch.nn.MSELoss()
    # # optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # , weight_decay=0.00001

    # data = dataset[0]
    # print(len(data[0][0]), torch.max(data[1]))
    # Audio(data[0], rate=22050, autoplay=False)


if __name__ == '__main__':
    main()
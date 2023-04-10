
from components import UNetV0
from models import DiffusionModel
from diffusion import VDiffusion, VSampler
from data_loader import AudioDataset
from torch.utils.data import Dataset, DataLoader
import torch

model = DiffusionModel(
    net_t=UNetV0,  # The model type used for diffusion (U-Net V0 in this case)
    in_channels=1,  # U-Net: number of input/output (audio) channels
    # U-Net: channels at each layer
    channels=[16, 32, 64, 128, 256, 512, 512, 1024, 1024],
    # U-Net: downsampling and upsampling factors at each layer
    factors=[1, 1, 1, 1, 1, 1, 1, 1, 1],
    # U-Net: number of repeating items at each layer
    items=[1, 2, 2, 2, 2, 2, 2, 4, 4],
    # U-Net: attention enabled/disabled at each layer
    attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1],
    attention_heads=8,  # U-Net: number of attention heads per attention item
    attention_features=64,  # U-Net: number of attention features per attention item
    diffusion_t=VDiffusion,  # The diffusion method used
    sampler_t=VSampler,  # The diffusion sampler used
)

# create the dataloader
audio_dir = "Data/DogBark"
dataset = AudioDataset(audio_dir)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Iterate over training data
for i, (inputs, _) in enumerate(dataloader, 0):

    # Forward pass and compute loss
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, inputs)

    # Backward pass and optimize
    loss.backward()
    optimizer.step()

    # Print statistics
    if i % 1 == 0:
        print(f"[{i+1}/{len(dataloader)}] Loss: {loss.item():.5f}")

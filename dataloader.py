import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import numpy as np
from IPython.display import Image, Audio
from torch.utils.data import Dataset, DataLoader
import os
import glob
from diffusion import Diffusion


class AudioDataset(Dataset):
    def __init__(self, file_path: str, device: torch.device, diffusion: Diffusion, length=88200, sample_rate=22050, ):
        files = glob.glob(os.path.join(
            file_path, "**", "*.wav"), recursive=True)

        self.labels = {
            1: 'DogBark',
            2: 'Sneeze_Cough',
            3: 'Rain',
            4: 'MovingMotorVehicle',
            5: 'Keyboard',
            6: 'GunShot',
            7: 'Footstep',
        }
        waveforms = []
        for label, folder in self.labels.items():
            dir_path = os.path.join(file_path, folder)
            files = glob.glob(os.path.join(dir_path, "*.wav"))
            for path in files:
                waveform, sr = torchaudio.load(path)
                waveform = waveform * 0.98 / torch.max(waveform)
                waveforms.append((label, waveform))

        self.waveforms = waveforms
        self.sample_rate = sample_rate
        self.device = device
        self.length = length
        self.diffusion = diffusion

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        label, waveform = self.waveforms[idx]
        waveform = waveform.to(self.device)

        # Apply gain
        waveform = waveform * np.random.uniform(0.75, 1)

        # create a different starting point and roll the data over
        waveform = torch.roll(waveform, np.random.randint(waveform.shape[0]))

        # create the diffusion
        timestamp = np.random.randint(1, 1000)
        x_t, noise = self.diffusion(waveform, timestamp)

        x_t = x_t.to(self.device)
        noise = noise.to(self.device)
        timestamp = torch.ones(1, device=self.device) * timestamp
        label = torch.ones(1, device=self.device) * label

        return x_t, noise, timestamp, label

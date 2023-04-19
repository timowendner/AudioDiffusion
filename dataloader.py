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

        waveforms = []
        for path in files:
            waveform, sr = torchaudio.load(path)
            waveform = waveform * 0.98 / torch.max(waveform)
            waveforms.append(waveform)

        self.waveforms = torch.stack(waveforms)
        self.sample_rate = sample_rate
        self.device = device
        self.length = length
        self.diffusion = diffusion

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        waveform = self.waveforms[idx].to(self.device)

        # Apply gain
        waveform = waveform * np.random.uniform(0.7, 1)

        # create a different starting point and roll the data over
        waveform = torch.roll(waveform, np.random.randint(waveform.shape[0]))

        # create the diffusion
        t = np.random.randint(1, 1000)
        x_t, noise = self.diffusion(waveform, t)

        x_t = x_t.to(self.device)
        noise = noise.to(self.device)
        t = torch.ones(1, device=self.device) * t
        label = torch.ones(1, device=self.device) * 1

        return x_t, noise, t, label

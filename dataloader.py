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
    def __init__(self, diffusion: Diffusion, config, device: torch.device):
        waveforms = []
        for label, folder in config.label_path.items():
            label = int(label)
            if label not in config.label_train:
                continue
            dir_path = os.path.join(config.data_path, folder)
            files = glob.glob(os.path.join(dir_path, "*.wav"))
            for path in files:
                waveform, sr = torchaudio.load(path)
                waveform = waveform * 0.98 / torch.max(waveform)
                waveforms.append((label+1, waveform))

        if len(waveforms) == 0:
            raise AttributeError('Data-path seems to be empty')

        self.waveforms = waveforms
        self.device = device
        self.length = config.audio_length
        self.diffusion = diffusion

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        label, waveform = self.waveforms[idx]
        waveform = waveform.to(self.device)

        # Apply gain
        waveform = waveform * (1 - np.random.normal(0, 0.15)**2)

        # create a different starting point and roll the data over
        waveform = torch.roll(waveform, np.random.randint(waveform.shape[0]))

        # create the diffusion
        max_timestamp = self.diffusion.steps
        timestamp = np.random.randint(1, max_timestamp)
        x_t, noise = self.diffusion(waveform, timestamp)

        x_t = x_t.to(self.device)
        noise = noise.to(self.device)

        return x_t, noise, timestamp, label

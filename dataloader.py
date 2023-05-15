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
        """
        Initializes the AudioDataset class, which represents an audio dataset.

        Args:
            diffusion (Diffusion): the diffusion object used for adding noise
            config (Config): the configuration object for the model
            device (torch.device): the device to use for loading the data
        """
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
        """
        Retrieves an item from the dataset, with gain and rollover.

        Args:
            idx (int): the index of the item to retrieve

        Returns:
            Tuple[Tensor, Tensor, int, int]:
                x_t (torch.Tensor): The audio-file with added noise.
                noise (torch.Tensor): The noise that was added to the audio-file.
                timestamp (int): the current timestamp
                label (int): the current label
        """
        label, waveform = self.waveforms[idx]
        waveform = waveform.to(self.device)

        # Apply gain
        waveform = waveform * (1 - np.random.normal(0, 0.15)**2)

        # create a different starting point and roll the data over
        start = np.random.randint(waveform.shape[1])
        waveform = torch.roll(waveform, start, dims=1)

        # create the diffusion
        timestamp = np.random.randint(1, self.diffusion.steps)
        x_t, noise = self.diffusion(waveform, timestamp)

        x_t = x_t.to(self.device)
        noise = noise.to(self.device)

        return x_t, noise, timestamp, label

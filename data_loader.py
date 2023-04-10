import torch
import torchaudio
import torchaudio.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os


class AudioDataset(Dataset):
    def __init__(self, audio_dir):
        self.audio_dir = audio_dir
        self.audio_files = os.listdir(audio_dir)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # load the data
        audio_file = os.path.join(self.audio_dir, self.audio_files[idx])
        waveform, sr = torchaudio.load(audio_file)
        waveform = torch.mean(waveform, dim=1, keepdim=True)
        print(waveform.shape, sr)

        # specify the probabilities of that data augmentation
        eq_probability = 0.3
        gain_probability = 0.3

        # Apply EQ
        if np.random.random() < eq_probability:
            waveform = F.equalizer_biquad(
                waveform,
                sr,
                center_freq=np.random.choice([20, 60, 100, 250, 400, 630, 900, 1200, 1600,
                                             2000, 2500, 3150, 4000, 5000,
                                             6300, 8000, 10000, 12500, 16000, 20000]),
                gain=np.random.uniform(-6, 6),
                Q=0.707
            )

        # Apply gain
        if np.random.random() < gain_probability:
            gain = np.random.uniform(-6, 6)
            Vol = torchaudio.transforms.Vol(1)
            waveform = Vol(waveform)

        return waveform, sr

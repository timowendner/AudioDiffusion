import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import numpy as np
from IPython.display import Image, Audio
from torch.utils.data import Dataset, DataLoader
import os
import glob


class AudioDataset(Dataset):
    def __init__(self, file_path: str, device: torch.device, sample_rate=32768):
        files = glob.glob(os.path.join(
            file_path, "**", "*.wav"), recursive=True)[:10]

        waveforms = []
        for path in files:
            waveform, sr = torchaudio.load(path)
            waveform = waveform * 0.98 / torch.max(waveform)
            waveforms.append(waveform)

        self.waveforms = torch.stack(waveforms)
        self.sample_rate = sample_rate
        self.device = device

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        waveform = self.waveforms[idx]
        sr = self.sample_rate

        # specify the probabilities of the data augmentation
        eq_probability = 0.3

        # Apply EQ
        # freq_bands = (20, 60, 100, 250, 400, 630, 900, 1200, 1600, 2000, 2500,
        #               3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000)
        # if torch.rand(1,) < eq_probability:
        #     waveform = F.equalizer_biquad(
        #         waveform,
        #         sr,
        #         center_freq=np.random.choice(freq_bands),
        #         gain=np.random.uniform(-6, 6),
        #         Q=0.707
        #     )

        # Apply gain
        # waveform = waveform * (1 - torch.rand(1,) ** 4)

        # create a different starting point and roll the data over
        start = torch.randint(waveform.shape[1], size=(1,))
        waveform = torch.roll(waveform, int(start))

        # create the diffusion
        # r = (torch.rand(1,) ** 2) * 0.0
        # noise = (torch.rand(waveform.shape,) * 2 - 1) * r
        # model_input = waveform + noise
        # noise = (torch.rand(waveform.shape,)
        #          * 2 - 1) * torch.clamp(r - 0.5, min=0)
        # target = waveform + noise

        model_input = model_input * 0.98 / torch.max(model_input)
        target = target * 0.98 / torch.max(target)

        model_input = torch.from_numpy(np.ones(model_input.shape)).float()
        target = torch.from_numpy(np.ones(target.shape)).float()
        print(waveform.shape, model_input.shape)

        return waveform.to(self.device), target.to(self.device)

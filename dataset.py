import musdb
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Tuple

class MUSDB18Dataset(Dataset):
    """
    Dataset MUSDB18
    """
    def __init__(self, base_path: str, subset: str, split: str, 
                 duration: Optional[float], nfft: int, samples: int = 1, 
                 random: bool = False, partitions: int = 1) -> None:
        """
        base_path -- Ruta del dataset
        subset -- Nombre del conjunto: 'train' o 'test'
        split -- División del entrenamiento: 'train' o 'valid' cuando subset='train'
        duration -- Duración de cada canción en segundos
        nfft -- Cantidad de puntos de la fft
        samples -- Cantidad de muestras de cada cancion
        random -- True si se van a mezclar las canciones de forma aleatoria
        partitions -- Cantidad de particiones de las canciones de validación
        """
        super(MUSDB18Dataset, self).__init__()
        self.sample_rate = 44100 # Por ahora no se usa
        self.split = split
        self.duration = duration
        self.nfft = nfft
        self.samples = samples
        self.random = random
        self.partitions = partitions
        self.window = torch.hann_window(nfft)
        self.stems = ['vocals', 'drums', 'bass', 'other']
        self.mus = musdb.DB(root=base_path, subsets=subset, split=split)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.split == 'train' and self.duration:
            sources = []
            track = self.mus[index // self.samples]
            orig_mix = torch.zeros(2, int(self.sample_rate * self.duration))

            for source in self.stems:
                if self.random:
                    track = random.choice(self.mus)

                track.chunk_duration = self.duration
                track.chunk_start = random.uniform(0, track.duration - self.duration)

                audio = track.sources[source].audio.T
                audio *= np.random.uniform(0.25, 1.25, (audio.shape[0], 1))
                if random.random() < 0.5:
                    audio = np.flipud(audio)
                audio = torch.as_tensor(audio.copy(), dtype=torch.float32)
                orig_mix += audio
                stft = torch.stft(audio, n_fft=self.nfft, window=self.window,
                                  onesided=True, return_complex=True)
                sources.append(stft)

            x = torch.stft(orig_mix, n_fft=self.nfft, window=self.window,
                           onesided=True, return_complex=True)
            y = torch.stack(sources, dim=0)

        # Validación y Test
        else:
            track = self.mus[index // self.partitions]

            chunk = track.duration // self.partitions
            track.chunk_start = (index % self.partitions) * chunk
            if (index + 1) % self.partitions == 0:
                track.chunk_duration = track.duration - track.chunk_start
            else:
                track.chunk_duration = chunk

            sources = []
            for source in self.stems:
                audio = track.sources[source].audio.T
                audio = torch.as_tensor(audio.copy(), dtype=torch.float32)
                stft = torch.stft(audio, n_fft=self.nfft, window=self.window,
                                  onesided=True, return_complex=True)
                sources.append(stft)

            x = torch.stft(track.audio.T, n_fft=self.nfft, window=self.window,
                           onesided=True, return_complex=True)
            y = torch.stack(sources, dim=0)
        return (torch.real(x), torch.imag(x)), (torch.real(y), torch.imag(y))

    def __len__(self) -> int:
        return len(self.mus) * self.samples * self.partitions
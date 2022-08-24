import os
from pathlib import Path
from typing import Tuple, Union
from torch import Tensor
from torch.utils.data import Dataset
import torchaudio


class BBC(Dataset):
    """
    Create the BBC dataset.

    Args:
        root (str or Path):
            Path to the directory where the dataset is found.
            It must include the metadata file "meta.tsv".
    """
    def __init__(
        self,
        root: Union[str, Path],
    ) -> None:
        super().__init__()

        self.root = os.fspath(root)
        assert os.path.isdir(root)

        self._tsv = os.path.join(root, "meta.tsv")
        rows = open(self._tsv, "r", encoding='utf-8').read()
        lines = rows.split("\n")
        self._meta = [k.split("\t") for k in lines[1:]]

    def __getitem__(self, i: int) -> Tuple[Tensor, int, set, str, str]:
        """Load the i-th sample from the dataset.

        Args:
            i (int): The index of the sample to be loaded.

        Returns:
            waveform, sample_rate, labels, mids (Tuple):
                mids are the Freebase identifiers corresponding to the class
                labels, as defined in the AudioSet Ontology specification
        """
        if i >= len(self):
            raise IndexError
        path_wav = os.path.join(self.root, self._meta[i][0])
        waveform, sample_rate = torchaudio.backend.sox_io_backend.load(path_wav)
        labels = self._meta[i][2:5]
        return waveform, sample_rate, *labels

    def __len__(self) -> int:
        return len(self._meta)

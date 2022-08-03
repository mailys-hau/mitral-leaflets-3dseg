import h5py
import torch

from itertools import accumulate # Faster than numpy if you manipulate list
from pathlib import Path
from torch import from_numpy as fnp
from torch.utils.data import Dataset



class HDFDataset(Dataset):
    """ Load from pre-processed HDF files """
    def __init__(self, data_dir, hdfnames, total_frames, norm=None):
        super(HDFDataset, self).__init__()
        self.prefix = Path(data_dir).expanduser().resolve()
        self._setup_helpers(hdfnames)
        self.length = total_frames
        #FIXME: Define default norm
        #FIXME? Use a full transform instead of just norm
        self.norm = norm #Expect callable

    def _setup_helpers(self, hdfnames):
        self.fnames = []
        self.sequences_indexes = []
        nb_frame_by_seq = []
        for i, d in enumerate(hdfnames):
            self.fnames.append(d[0])
            self.sequences_indexes += d[1] * [i]
            nb_frame_by_seq.append(d[1])
        self.cumulative_frame_len = list(accumulate(nb_frame_by_seq, initial=0))


    def _load_volume(self, i):
        #FIXME: Handle negative index
        seq_idx = self.sequences_indexes[i]
        frame_idx = i - self.cumulative_frame_len[seq_idx] + 1
        #FIXME: Bottleneck, find a way to bufferise some frames
        hdfile = h5py.File(self.prefix.joinpath(self.fnames[seq_idx]), 'r')
        vin = fnp(hdfile["CartesianVolumes"][f"vol{frame_idx:02d}"][()]).to(torch.float)
        vout = fnp(hdfile["Labels"][f"vol{frame_idx:02d}"][()]).to(torch.long)
        hdfile.close()
        return vin, vout


    def __getitem__(self, i):
        return self._load_volume(i)
        #return self.norm(vol)

    def __len__(self):
        return self.length

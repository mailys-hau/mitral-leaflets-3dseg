import h5py
import torch
import torch.nn as nn

from itertools import accumulate # Faster than numpy if you manipulate list
from pathlib import Path
from torch import from_numpy as fnp
from torch.utils.data import Dataset



class HDFDataset(Dataset):
    """ Load frame by frame from pre-processed HDF files """
    def __init__(self, data_dir, hdfnames, total_frames,
                 norm=lambda x: x / 256, multiclass=False, cache=False):
        super(HDFDataset, self).__init__()
        self.prefix = Path(data_dir).expanduser().resolve()
        self._setup_helpers(hdfnames)
        self.nb_frames = total_frames
        #FIXME? Default norm as identity
        #FIXME? Use a full transform instead of just norm
        self.norm = norm #Expect callable
        self.multiclass = multiclass
        self.cache = {} if cache else None

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
        hdfile = h5py.File(self.get_path(seq_idx), 'r')
        vin = fnp(hdfile["CartesianVolumes"][f"vol{frame_idx:02d}"][()]).to(torch.float)
        ant = fnp(hdfile["Labels"][f"ant{frame_idx:02d}"][()]).to(torch.bool)
        post = fnp(hdfile["Labels"][f"ant{frame_idx:02d}"][()]).to(torch.bool)
        hdfile.close()
        if self.multiclass:
            #FIXME: Some voxel are in both ant & post class
            none = ~(ant | post)
            #vout = nn.ModuleList([none, ant, post])
            vout = torch.stack([none, ant, post])
        else:
            leaflet = (ant | post)
            # This way is easier to handle both multiclass and binary class
            #vout = nn.ModuleList([~leaflet, leaflet])
            vout = torch.stack([~leaflet, leaflet])
        return vin, vout

    def get_path(self, i):
        return self.prefix.joinpath(self.fnames[i])

    def get_sequences(self): # Assemble frames to DICOM's sequence
        prev_nbf = 0 # PREVious NumBer of Frames
        for i in range(self.nb_sequences):
            vins, vouts = [], []
            # NumBer of Frames
            nbf = self.cumulative_frame_len[i + 1] - self.cumulative_frame_len[i]
            for j in range(nbf):
                tmp = self._load_volume(i + j)
                vins.append(tmp[0]), vouts.append(tmp[1])
            yield vins, vouts
            prev_nbf = nbf


    def __getitem__(self, i):
        # If you run on a big enough machine, take advantage of it :3
        if self.cache is None or i not in self.cache.keys():
            vin, vout = self._load_volume(i)
            vin = self.norm(vin) if self.norm else vin
            # Gray scale, i.e. 1 channel, need float to compute loss
            vin, vout = vin.unsqueeze(0), vout.to(torch.float)
            if self.cache is None:
                return vin, vout
            self.cache[i] = (vin, vout)
        return self.cache[i]

    def __len__(self):
        return self.nb_frames

    @property
    def nb_sequences(self):
        return len(self.fnames)


class DummyDataset(Dataset):
    def __init__(self, nb_dummies):
        size = 256 # Size of volume
        nc = 2 # Number of classes
        self.inputs = torch.rand(nb_dummies, size, size, size).to(torch.float)
        self.outputs = torch.randint(0, 2, (nb_dummies, nc, size, size, size)).to(torch.float)
        self.nb_dummies = nb_dummies

    def __getitem__(self, i):
        return self.inputs[i].unsqueeze(0), self.outputs[i]

    def __len__(self):
        return self.nb_dummies

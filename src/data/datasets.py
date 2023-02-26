import h5py
import monai.transforms as mt
import torch
import torch.nn as nn

from itertools import accumulate # Faster than numpy if you manipulate list
from pathlib import Path
from torch import from_numpy as fnp
from torch.utils.data import Dataset

from data.transforms import NORMS, RESIZE



class HDFDataset(Dataset):
    """ Load frame by frame from pre-processed HDF files """
    def __init__(self, data_dir, hdfnames, total_frames,
                 resize="by-classes", spatial_size=[128, 128, 128],
                 norm="256", contrast=None, multiclass=False, cache=False,
                 augmentation=False):
        super(HDFDataset, self).__init__()
        keys = ["in", "out"]
        self.prefix = Path(data_dir).expanduser().resolve()
        self._setup_helpers(hdfnames)
        self.nb_frames = total_frames
        self.resize = RESIZE[resize](keys, spatial_size, multiclass=multiclass)
        self.norm = NORMS[norm]
        self.multiclass = multiclass
        self.augmentation = self._get_augment(keys) if augmentation else False
        self.cache = {} if cache else None
        self.contrast = mt.AdjustContrast(contrast) if contrast is not None else None

    def _setup_helpers(self, hdfnames):
        self.fnames = []
        self.sequences_indexes = []
        nb_frame_by_seq = []
        for i, d in enumerate(hdfnames):
            self.fnames.append(d[0])
            self.sequences_indexes += d[1] * [i]
            nb_frame_by_seq.append(d[1])
        self.cumulative_frame_len = list(accumulate(nb_frame_by_seq, initial=0))

    def _get_augment(self, keys):
        return mt.Compose([
            # Move around (input, target)
            mt.RandRotated(keys, range_x=5, range_y=5, range_z=5),
            mt.RandAxisFlipd(keys),
            # Add noise to input
            mt.RandGaussianNoised(keys[0]),
            mt.RandGridDistortiond(keys[0])
            #mt.RandGridDistortiond(keys),
            # And many more...
        ])


    def _transform(self, inp, out, transform):
        data = mt.apply_transform(transform, {"in": inp, "out": out})
        return data["in"], data["out"]


    def _load_volume(self, i):
        #FIXME: Handle negative index
        seq_idx = self.sequences_indexes[i]
        frame_idx = i - self.cumulative_frame_len[seq_idx] + 1
        #FIXME: Bottleneck, find a way to bufferise some frames
        hdfile = h5py.File(self.get_path(seq_idx), 'r')
        vin = fnp(hdfile["CartesianVolume"][f"vol{frame_idx:02d}"][()])
        ant = hdfile["GroundTruth"][f"anterior-{frame_idx:02d}"][()]
        post = hdfile["GroundTruth"][f"posterior-{frame_idx:02d}"][()]
        ant, post = fnp(ant).to(torch.bool), fnp(post).to(torch.bool)
        hdfile.close()
        if self.multiclass:
            #FIXME: Some voxel are in both ant & post class
            none = ~(ant | post)
            vout = torch.stack([none, ant, post])
        else:
            leaflet = (ant | post)
            # This way is easier to handle both multiclass and binary class
            vout = torch.stack([~leaflet, leaflet])
        return self.norm(vin), vout

    def get_path(self, i):
        return self.prefix.joinpath(self.fnames[i])

    def get_sequences(self): # Assemble frames to DICOM's sequence
        prev_nbf = 0 # PREVious NumBer of Frames
        for i in range(self.nb_sequences):
            vins, vouts = [], []
            # NumBer of Frames
            nbf = self.cumulative_frame_len[i + 1] - self.cumulative_frame_len[i]
            for j in range(nbf):
                vin, vout = self[i + j] # Resize should be "center" for test
                vins.append(vin), vouts.append(vout)
            yield vins, vouts
            prev_nbf = nbf


    def __getitem__(self, i):
        # If you run on a big enough machine, take advantage of it :3
        if self.cache is not None and i in self.cache.keys():
            vin, vout = self.cache[i]
        else:
            vin, vout = self._load_volume(i)
            # Gray scale, i.e. 1 channel, need float to compute loss
            vin, vout = vin.unsqueeze(0), vout.to(torch.float)
            if self.contrast is not None:
                vin = self.contrast(vin)
            if self.cache is not None:
                self.cache[i] = (vin, vout)
        if self.augmentation: # Random so don't cache it
            vin, vout = self._transform(vin, vout, self.augmentation)
        # Can be random, so don't cache it
        return self._transform(vin, vout, self.resize)

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

import h5py

import monai.transforms as mt
import torch

from itertools import accumulate # Faster than numpy if you manipulate list
from pathlib import Path
from torch import from_numpy as fnp
from torch.utils.data import Dataset

from data.transforms import NORMS, RESIZE



class _HDFDataset(Dataset):
    """ Load volume from HDF using specific architecture """
    def __init__(self, data_dir, hdfnames, multiclass=False, 
                 resize="center-random", spatial_size=[128, 128, 128],
                 norm="256", contrast=None, augmentation=False, cache=False):
        super(_HDFDataset, self).__init__()
        self.prefixes = self._setup_prefixes(data_dir)
        self._setup_indexes(hdfnames)
        self.multiclass = multiclass
        keys = ["in", "out"]
        self.resize = RESIZE[resize](keys, spatial_size, multiclass=multiclass)
        self.norm = NORMS[norm]
        self.contrast = mt.AdjustContrast(contrast) if contrast is not None\
                            else contrast
        self.augmentation = self._define_augmentations(keys) if augmentation\
                                else augmentation
        self.cache = {} if cache else None

    def _setup_prefixes(self, prefixes):
        if isinstance(prefixes, list):
            return [ Path(p).expanduser().resolve() for p in prefixes ]
        return [ Path(prefixes).expanduser().resolve() ]

    def _setup_indexes(self, hdfnames):
        # Bunch of indexes' lists to ease __getitem__'s process
        self.fnames = []
        self.sequence_indexes = [] # Which sequence regarding the whole dataset
        nbf_per_seq = []# Number of frame per sequences
        for i, d in enumerate(hdfnames):
            self.fnames.append(d[0])
            self.sequence_indexes += d[1] * [i]
            nbf_per_seq.append(d[1])
        self.cumulative_nbf = list(accumulate(nbf_per_seq, initial=0))

    def _define_augmentations(self, keys):
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


    def do_transform(self, inp, out, transform):
        data = mt.apply_transform(transform, {"in": inp, "out": out})
        return data["in"], data["out"]


    def _load_volumes(self, iseq, iframe):
        #FIXME: Handle negative index
        #FIXME: Bottleneck if no cache
        hdfile = h5py.File(self.get_path(iseq), 'r')
        iframe += 1 # Indexes start at 1 in HDF
        vin = fnp(hdfile["CartesianVolume"][f"vol{iframe:02d}"][()])
        ant = hdfile["GroundTruth"][f"anterior-{iframe:02d}"][()]
        post = hdfile["GroundTruth"][f"posterior-{iframe:02d}"][()]
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

    def get_path(self, i): # Not the prettiest
        for p in self.prefixes:
            if p.joinpath(self.fnames[i]).is_file():
                return p.joinpath(self.fnames[i])

    def get_volumes(self, i, iseq, iframe):
        # i is the general index of the dataset
        # If you run on a big enough machine, take advantage of it :3
        if self.cache is not None and i in self.cache.keys():
            vin, vout = self.cache[i]
        else:
            vin, vout = self._load_volumes(iseq, iframe)
            # Gray scale, i.e. 1 channel, need float to compute loss
            vin, vout = vin.unsqueeze(0), vout.to(torch.float)
            if self.contrast is not None:
                vin = self.contrast(vin)
            if self.cache is not None:
                self.cache[i] = (vin, vout)
        if self.augmentation: # Random so don't cache it
            vin, vout = self.do_transform(vin, vout, self.augmentation)
        # Can be random, so don't cache it
        return self.do_transform(vin, vout, self.resize)

    @property
    def nb_sequences(self):
        return len(self.fnames)

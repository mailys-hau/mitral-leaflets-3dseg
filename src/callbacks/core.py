import h5py
import torch

from pathlib import Path
from pytorch_lightning.callbacks import Callback

from data.datasets.core import _ListHDFDataset



class EnhancedCallback(Callback):
    """ Add a few functionnalities that are going to be used repeatedly in this project """
    def __init__(self):
        super(EnhancedCallback, self).__init__()

    def get_voxinfo(self, fname):
        # Get additionnal informations on voxel grid
        # As all info are the same for all frames of input, predictions and 
        # annotations, we extract it separately to not open the file repeatedly
        hdf = h5py.File(fname, 'r')
        origin = hdf["VolumeGeometry"]["origin"][()]
        directions = hdf["VolumeGeometry"]["directions"][()]
        spacing = hdf["VolumeGeometry"]["resolution"][()]
        hdf.close()
        # Same order as required for VoxelGrid
        return origin, directions, spacing

    def resolve_dirpath(self, trainer, category):
        """ Solve saving directory if not default """
        if self.dirpath is not None:
            try:
                self.dirpath = Path(self.dirpath)
            except AttributeError as error:
                raise ValueError(f"{self.__class__.__name__} is not a storing callback. You shouldn't use `resolve_dirpath`")
        else:
            # Similar to `pl.callbacks.ModelCheckpoint` just use WandB "human-
            # readable run name" to make it easier to associate a directory to a run
            save_dir = Path(trainer.default_root_dir)
            if len(trainer.loggers) > 0:
                if trainer.loggers[0].save_dir is not None:
                    save_dir = Path(trainer.loggers[0].save_dir)
                name = trainer.loggers[0].name
                # Where it really differs from parent's class
                # We use the run's name for readibility purpose, and append the
                # ID to make sure the directory's name is unique
                run_name = trainer.loggers[0].experiment.name
                version = trainer.loggers[0].version # = WandB ID by default
                save_dir = save_dir.joinpath(str(name), f"{run_name}_{version}")
            self.dirpath = save_dir.joinpath(category)
        self.dirpath = self.dirpath.expanduser().resolve()
        self.dirpath.mkdir(parents=True, exist_ok=True)

    def rm_background(self, dataset, tgs, preds):
        # Target is given as (C, W, H, D) and predictions as (B, C, W, H, D) with B == 1
        if isinstance(dataset, _ListHDFDataset): # We're dealing with TensorList
            # Remove background
            tg = [ torch.cat([ t[1:] for t in tltg ]) for tltg in tgs ]
            #FIXME: Why are pred nested?
            pred = [ torch.cat([ p[0, 1:] for p in tlp ]) for tlp in preds[0] ]
        else:
            # Remove background and select proper frames
            tg = [ t[1:] for t in tgs ]
            pred = [ p[0, 1:] for p in preds ]
        return tg, pred

import h5py
import echoviz as ecv

from callbacks.core import EnhancedCallback



class _Plot(EnhancedCallback):
    def __init__(self, dirpath=None):
        super(_Plot, self).__init__()
        self.dirpath = dirpath

    def get_voxinfo(self, fname):
        # Get additionnal informations to plot voxel grid
        # As all info are the same for all frames of input, predictions and 
        # annotations, we extract it separately to not open the file repeatedly
        hdf = h5py.File(fname, 'r')
        origin = hdf["ImageGeometry"]["origin"][()]
        directions = hdf["ImageGeometry"]["directions"][()]
        spacing = hdf["ImageGeometry"]["voxelsize"][()]
        hdf.close()
        # Same order as required for ecv.VoxelGrid
        return origin, directions, spacing

    def t2v(self, tensors, voxinfo):
        # Convert tensor (or list of tensor) to `ecv.VoxelGrid` (or list of it)
        def _to_voxel(tensor, voxinfo):
            if tensor.ndim > 3 and tensor.shape[0] > 1: # i.e. multiclass
                return [ ecvVoxelGrid(t.squeeze(), *voxinfo) for t in tensor ]
            return ecv.VoxelGrid(tensor.squeeze(), *voxinfo)
        return _to_voxel(tensors, voxinfo) if not isinstance(tensors, list) \
                else [_to_voxel(t, voxinfo) for t in tensors]

    def to_dict(self, data):
        # Get ready before calling echoviz, some data need to be dict like
        if isinstance(data[0], list):
            return {"anterior": [d[0] for d in data],
                    "posterior": [d[1] for d in data]}
        return {"all": data}


class Plot4D(_Plot):
    def __init__(self, dirpath=None):
        super(Plot4D, self).__init__(dirpath)
        self.plotter = ecv.animated_3d

    def setup(self, trainer, pl_module, stage):
        self.resolve_dirpath(trainer, "plots_4d")

    def on_predict_epoch_end(self, trainer, pl_module, outputs):
        #FIXME: Handle multiclass
        dataset = trainer.predict_dataloaders[0].dataset
        preds = outputs[0]
        prev_nbf = 0
        for i, (inp, tg) in enumerate(dataset.get_sequences()):
            # Convert everything to `ecv.VoxelGrid`
            voxinfo = self.get_voxinfo(dataset.get_path(i))
            vin, vtg = self.t2v(inp, voxinfo), self.t2v(tg, voxinfo)
            nbf = len(vin)
            # First is False class, we don't plot it
            vpred = [ p[:,1:] for p in preds[prev_nbf:nbf] ]
            vpred = self.t2v(vpred, voxinfo)
            # Set it to dict for `echoviz` to read it
            vtg, vpred = self.to_dict(vtg), self.to_dict(vpred)
            # Plot & save
            name = dataset.fnames[i]
            fname = self.dirpath.joinpath(name).with_suffix(".png")
            #FIXME? Add train ID in plot's title
            self.plotter(vin, vtg, vpred, title=f"{name}'s result",
                         show=False, filename=fname)
            prev_nbf = nbf

    def on_train_epoch_end(self, trainer, pl_module): # Skip?
        pass
    def on_validation_epoch_end(self, trainer, pl_module):
        pass

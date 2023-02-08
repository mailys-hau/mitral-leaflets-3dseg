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
                return [ ecv.VoxelGrid(t.squeeze(), *voxinfo) for t in tensor ]
            return ecv.VoxelGrid(tensor.squeeze(), *voxinfo)
        return _to_voxel(tensors, voxinfo) if not isinstance(tensors, list) \
                else [_to_voxel(t, voxinfo) for t in tensors]

    def to_dict(self, data):
        # Get ready before calling echoviz, some data need to be dict like
        if isinstance(data[0], list):
            return {"anterior": [d[0] for d in data],
                    "posterior": [d[1] for d in data]}
        return {"all": data}

    def on_predict_epoch_end(self, trainer, pl_module, outputs):
        #FIXME: Handle multiclass
        dataset = trainer.predict_dataloaders[0].dataset
        preds = outputs[0]
        prev_nbf = 0
        for i, (inp, tg) in enumerate(dataset.get_sequences()):
            # Convert everything to `ecv.VoxelGrid`
            voxinfo = self.get_voxinfo(dataset.get_path(i))
            vin = self.t2v(inp, voxinfo)
            # First class is background, we don't plot it
            nbf = len(vin)
            vtg = [ t[1:] for t in tg ]
            # preds is given as (B,C,W,H,D) and B = 1
            vpred = [ p[0,1:] for p in preds[prev_nbf:nbf + prev_nbf] ]
            vtg, vpred = self.t2v(vtg, voxinfo), self.t2v(vpred, voxinfo)
            name = dataset.fnames[i]
            fname = self.dirpath.joinpath(name)
            # Go down to children to plot, since some take a full sequence and
            # other just a volume as input
            self.do_plot(vin, vtg, vpred, fname)
            prev_nbf = nbf


class Plot4D(_Plot):
    def __init__(self, dirpath=None):
        super(Plot4D, self).__init__(dirpath)
        self.plotter = ecv.animated_3d

    def setup(self, trainer, pl_module, stage):
        self.resolve_dirpath(trainer, "plots_4d")

    def do_plot(self, vinputs, vlabels, vpreds, fname):
        # Set it to dict for `echoviz` to read it
        fname = fname.with_suffix(".html")
        vtg, vpred = self.to_dict(vlabels), self.to_dict(vpreds)
        #FIXME? Add train ID in plot's title
        self.plotter(vinputs, vtg, vpred, title=f"{fname.stem}'s result",
                     show=False, filename=fname)


class Plot4DSlice(_Plot):
    def __init__(self, index=128, axis=1, dirpath=None):
        super(Plot4DSlice, self).__init__(dirpath)
        self.plotter = ecv.sliced_sequence
        self.index = index
        self.axis = axis

    def setup(self, trainer, pl_module, stage):
        self.resolve_dirpath(trainer, "plots_4d_sliced")

    def do_plot(self, vinputs, vlabels, vpreds, fname):
        # Set it to dict for `echoviz` to read it
        vtg, vpred = self.to_dict(vlabels), self.to_dict(vpreds)
        #FIXME? Add train ID in plot's title
        fname = fname.with_suffix(".gif")
        self.plotter(vinputs, vtg, self.index, self.axis, vpred,
                     title=f"{fname.stem}'s result", filename=fname)

class Plot3DSlice(_Plot):
    def __init__(self, frame_stride=1, slice_stride=10, axis=1, dirpath=None):
        super(Plot3DSlice, self).__init__(dirpath)
        self.plotter = ecv.sliced_volume
        self.frame_stride = frame_stride
        self.slice_stride = slice_stride
        self.axis = axis

    def setup(self, trainer, pl_module, stage):
        self.resolve_dirpath(trainer, "plots_3d_sliced")

    def to_dict(self, data):
        # Get ready before calling echoviz, some data need to be dict like
        if isinstance(data, list):
            return {"anterior": data[0], "posterior": data[1]}
        return {"all": data}

    def do_plot(self, vinputs, vlabels, vpreds, fname):
        # Receive voxels of a full sequence
        for i in range(0, len(vinputs), self.frame_stride):
            vin, vlabel, vpred = vinputs[i], vlabels[i], vpreds[i]
            vtg, vpred = self.to_dict(vlabel), self.to_dict(vpred)
            filename = fname.with_stem(fname.stem + f"-frame-{i:02d}").with_suffix(".gif")
            #FIXME? Add train ID in plot's title
            self.plotter(vin, vtg, self.axis, vpred, stride=self.slice_stride,
                         title=f"{fname.stem}'s frame {i:02d}", filename=filename)

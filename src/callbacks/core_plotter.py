from callbacks.core import EnhancedCallback
from echoviz import VoxelGrid



class Plotter(EnhancedCallback):
    """ Additionnal functionnalities used for plotting callbacks """
    def __init__(self, dirpath=None):
        super(Plotter, self).__init__()
        self.dirpath = dirpath
        self.by_frame = False

    def t2v(self, tensors, voxinfo):
        # Convert tensor (or list of tensor) to `VoxelGrid` (or list of it)
        def _to_voxel(tensor, voxinfo):
            if tensor.ndim > 3 and tensor.shape[0] > 1: # i.e. multiclass
                return [ VoxelGrid(t.squeeze(), *voxinfo) for t in tensor ]
            return VoxelGrid(tensor.squeeze(), *voxinfo)
        return _to_voxel(tensors, voxinfo) if not isinstance(tensors, list) \
                else [_to_voxel(t, voxinfo) for t in tensors]

    def to_dict(self, data):
        # Get ready before calling echoviz, some data need to be dict like
        if isinstance(data, list):
            if self.by_frame or isinstance(data[0], VoxelGrid):
                return {"anterior": data[0], "posterior": data[1]}
            return {"anterior": [d[0] for d in data],
                    "posterior": [d[1] for d in data]}
        return {"all": data}

    def on_predict_epoch_end(self, trainer, pl_module, outputs):
        # We have to wait until the end of the epoch to get the sequences
        dataset = trainer.predict_dataloaders[0].dataset
        preds = outputs[0] # We get all sequences at once
        prev_nbf = 0
        for iseq in range(dataset.nb_sequences):
            inp, tg = dataset.get_sequence(iseq)
            # Convert everything to `VoxelGrid`
            voxinfo = self.get_voxinfo(dataset.get_path(iseq))
            vin = self.t2v(inp, voxinfo)
            nbf = len(vin)
            # First class is background, we don't plot it
            vtg = [ self.t2v(t[1:], voxinfo) for t in tg ]
            # preds is given as (B,C,W,H,D) and B = 1
            vpred = [ self.t2v(p[0,1:], voxinfo) for p in preds[prev_nbf:nbf + prev_nbf] ]
            fname = self.dirpath.joinpath(dataset.fnames[iseq])
            # Go down to children to plotter for the detail
            self.do_plot(vin, vtg, vpred, fname)
            prev_nbf = nbf


class SlicePlotter(Plotter):
    def __init__(self, axis=0, dirpath=None):
        super(SlicePlotter, self).__init__(dirpath)
        self.axis = axis

    @property
    def view(self):
        if self.axis == 0: # Long axis view
            return "lax"
        elif self.axis == 1: # Perpendicular to long axis view
            return "plax"
        return "sax" # Short axis view

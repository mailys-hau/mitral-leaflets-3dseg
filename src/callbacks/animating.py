import h5py
import echoviz as ecv

from callbacks.core_plotter import Plotter, SlicePlotter



class Plot4D(Plotter):
    def __init__(self, dirpath=None):
        super(Plot4D, self).__init__(dirpath)
        self.plotter = ecv.animated_3d

    def setup(self, trainer, pl_module, stage):
        self.resolve_dirpath(trainer, "4Dplots")

    def do_plot(self, vinputs, vlabels, vpreds, fname):
        if len(vinputs) == 1: # Most likely FrameDataset
            return UserWarning("Received a single volume, not plotting. Use `callbacks.Plot3D` instead.")
        fname = fname.with_suffix(".html")
        # Set it to dict for `echoviz` to read it
        vtg, vpred = self.to_dict(vlabels), self.to_dict(vpreds)
        self.plotter(vinputs, vtg, vpred, title=f"{fname.stem}'s result",
                     show=False, filename=fname)


class SliceSequencePlot(SlicePlotter):
    def __init__(self, index=44, axis=0, dirpath=None):
        super(SliceSequencePlot, self).__init__(axis, dirpath)
        self.plotter = ecv.sliced_sequence
        self.index = index

    def setup(self, trainer, pl_module, stage):
        self.resolve_dirpath(trainer, f"seq-slice_{self.view}")

    def do_plot(self, vinputs, vlabels, vpreds, fname):
        # Set it to dict for `echoviz` to read it
        if len(vinputs) == 1: # Most likely FrameDataset
            return UserWarning("Received a single volume, not plotting. Use `callbacks.SlicePlot` instead.")
        vtg, vpred = self.to_dict(vlabels), self.to_dict(vpreds)
        fname = fname.with_suffix(".gif")
        self.plotter(vinputs, vtg, self.index, self.axis, vpred,
                     title=f"{fname.stem}'s result", filename=fname)

class SliceVolumePlot(SlicePlotter):
    def __init__(self, frame_stride=1, slice_stride=10, axis=0, dirpath=None):
        super(SliceVolumePlot, self).__init__(axis, dirpath)
        self.plotter = ecv.sliced_volume
        self.frame_stride = frame_stride
        self.slice_stride = slice_stride
        self.by_frame = True # You only plot one frame at a time

    def setup(self, trainer, pl_module, stage):
        self.resolve_dirpath(trainer, f"vol-slice_{self.view}")

    def do_plot(self, vinputs, vlabels, vpreds, fname):
        fname = fname.with_suffix(".gif")
        if len(vinputs) == 1: # Most likely FrameDataset
            vin, vlabel, vpred = vinputs[0], vlabels[0], vpreds[0]
            # Set it to dict for `echoviz` to read it
            vtg, vpred = self.to_dict(vlabel), self.to_dict(vpred)
            self.plotter(vin, vtg, self.axis, vpred, stride=self.slice_stride,
                         title=f"{fname.stem}'s result", filename=fname)
            return
        # Receive voxels of a full sequence
        for i in range(0, len(vinputs), self.frame_stride):
            vin, vlabel, vpred = vinputs[i], vlabels[i], vpreds[i]
            # Set it to dict for `echoviz` to read it
            vtg, vpred = self.to_dict(vlabel), self.to_dict(vpred)
            filename = fname.with_stem(fname.stem + f"-frame-{i:02d}")
            self.plotter(vin, vtg, self.axis, vpred, stride=self.slice_stride,
                         title=f"{fname.stem}'s frame {i:02d}",
                         filename=filename)

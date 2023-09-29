import echoviz as ecv

from callbacks.core_plotter import Plotter, SlicePlotter



class Plot3D(Plotter):
    def __init__(self, frame_stride=1, dirpath=None):
        super(Plot3D, self).__init__(dirpath)
        self.plotter = ecv.interactive_3d
        self.frame_stride = frame_stride
        self.by_frame = True

    def setup(self, trainer, pl_module, stage):
        self.resolve_dirpath(trainer, "3Dplots")

    def do_plot(self, vinputs, vlabels, vpreds, fname):
        fname = fname.with_suffix(".html")
        if len(vinputs) == 1: # When using the FrameDataset
            # If one frame, don't add its number to filename
            vin, vtg, vpred = vinputs[0], vlabels[0], vpreds[0]
            # Set it to dict for `echoviz` to read it
            vtg, vpred = self.to_dict(vtg), self.to_dict(vpred)
            self.plotter(vin, vtg, vpred, title=f"{fname.stem}'s result", show=False,
                         filename=fname)
            return
        for i in range(0, len(vinputs), self.frame_stride):
            vin, vtg, vpred = vinputs[i], vlabels[i], vpreds[i]
            # Set it to dict for `echoviz` to read it
            vtg, vpred = self.to_dict(vtg), self.to_dict(vpred)
            filename = fname.with_stem(fname.stem + f"-frame-{i:02d}")
            self.plotter(vin, vtg, vpred, title=f"{fname.stem}'s frame {i:02d}",
                         show=False, filename=filename)

class SlicePlot(SlicePlotter):
    def __init__(self, frame_stride=1, index=44, axis=0, dirpath=None):
        super(SlicePlot, self).__init__(axis, dirpath)
        self.plotter = ecv.plot_slice
        self.frame_stride = frame_stride
        self.index = index
        self.by_frame = True

    def setup(self, trainer, pl_module, stage):
        self.resolve_dirpath(trainer, f"3Dplots_sliced_{self.view}")

    def do_plot(self, vinputs, vlabels, vpreds, fname):
        fname = fname.with_suffix(".png")
        if len(vinputs) == 1: # When using the FrameDataset
            # If one frame, don't add its number to filename
            vin, vtg, vpred = vinputs[0], vlabels[0], vpreds[0]
            # Set it to dict for `echoviz` to read it
            vtg, vpred = self.to_dict(vtg), self.to_dict(vpred)
            self.plotter(vin, vtg, self.index, self.axis, vpred,
                         title=f"{fname.stem}'s result", show=False, filename=fname)
            return
        for i in range(0, len(vinputs), self.frame_stride):
            vin, vtg, vpred = vinputs[i], vlabels[i], vpreds[i]
            # Set it to dict for `echoviz` to read it
            vtg, vpred = self.to_dict(vtg), self.to_dict(vpred)
            filename = fname.with_stem(fname.stem + f"-frame-{i:02d}")
            self.plotter(vin, vtg, self.index, self.axis, vpred,
                         title=f"{fname.stem}'s frame {i:02d}", show=False,
                         filename=filename)


class Plot3DDistance(Plotter):
    def __init__(self, distance, frame_stride=1, dirpath=None):
        super(Plot3DDistance, self).__init__(dirpath)
        if distance.lower() == "asd":
            self.plotter = ecv.asd_interactive_3d
        elif distance.lower() == "sdf":
            self.plotter = ecv.sdf_interactive_3d
        else:
            raise ValueError(f"Uknown distance '{distance}'.")
        self.distance = distance
        self.frame_stride = frame_stride
        self.by_frame = True

    def setup(self, trainer, pl_module, stage):
        self.resolve_dirpath(trainer, f"3Dplots_{self.distance}")

    def set_to_mm(self, vin, vtg, vpred):
        for i in range(len(vin)):
            vin[i].set_scale("mm")
            if isinstance(vtg[i], list):
                for j in range(len(vtg[i])):
                    vtg[i][j].set_scale("mm")
                    vpred[i][j].set_scale("mm")
            else:
                vtg[i].set_scale("mm")
                vpred[i].set_scale("mm")

    def do_plot(self, vinputs, vlabels, vpreds, fname):
        fname = fname.with_suffix(".html")
        self.set_to_mm(vinputs, vlabels, vpreds)
        if len(vinputs) == 1: # When using the FrameDataset
            # If one frame, don't add its number to filename
            vin, vtg, vpred = vinputs[0], vlabels[0], vpreds[0]
            # Set it to dict for `echoviz` to read it
            vtg, vpred = self.to_dict(vtg), self.to_dict(vpred)
            self.plotter(vtg, vpred, vinputs=vin, title=f"{fname.stem}'s result",
                         show=False, filename=fname)
            return
        for i in range(0, len(vinputs), self.frame_stride):
            vin, vtg, vpred = vinputs[i], vlabels[i], vpreds[i]
            # Set it to dict for `echoviz` to read it
            vtg, vpred = self.to_dict(vtg), self.to_dict(vpred)
            filename = fname.with_stem(fname.stem + f"-frame-{i:02d}")
            self.plotter(vtg, vpred, vinputs=vin, title=f"{fname.stem}'s frame {i:02d}",
                         show=False, filename=filename)

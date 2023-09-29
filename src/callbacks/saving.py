import h5py

from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint

from callbacks.core import EnhancedCallback



class EnhancedModelCheckpoint(ModelCheckpoint):
    """ Same as `pl.ModelCheckpoint` but use the human redeable name as
        directory to save checkpoints """
    def __init__(self,**kwargs):
                 #dirpath=None, filename=None, monitor=None, verbose=False,
                 #save_last=None, save_top_k=1, save_weights_only=False,
                 #mode="min", auto_insert_metric_name=True,
                 #every_n_train_steps=None, train_time_interval=None,
                 #every_n_epochs=None, save_on_train_epoch_end=None):
        super(EnhancedModelCheckpoint, self).__init__(**kwargs)


    def setup(self, trainer, pl_module, stage):
        # Needed so our __resolve_ckpt_dir is called
        self.dirpath = self.__resolve_ckpt_dir(trainer)
        super(EnhancedModelCheckpoint, self).setup(trainer, pl_module, stage)

    def __resolve_ckpt_dir(self, trainer):
        if self.dirpath is not None:
            save_dir = Path(self.dirpath).resolve().expanduser()
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
            save_dir = save_dir.joinpath("checkpoints")
        return save_dir.expanduser().resolve()


class SavePredictedSequence(EnhancedCallback):
    def __init__(self, dirpath=None):
        self.dirpath = dirpath

    def setup(self, trainer, pl_module, stage):
        self.resolve_dirpath(trainer, "predictions")

    def add_voxinfo(self, fname, hdf):
        origin, directions, spacing = self.get_voxinfo(fname)
        info = hdf.create_group("/VolumeGeometry")
        info.create_dataset("origin", data=origin)
        info.create_dataset("directions", data=directions)
        info.create_dataset("resolution", data=spacing)

    def on_predict_epoch_end(self, trainer, pl_module, outputs):
        #FIXME? Handle several dataloader
        #FIXME: Multiprocess + nice tqdm bar
        dataset = trainer.predict_dataloaders[0].dataset
        preds = outputs[0]
        prev_nbf = 0
        for iseq in range(dataset.nb_sequences):
            inp, tg = dataset.get_sequence(iseq)
            nbf = len(inp)
            # Remove background and select proper frames
            tg, pred = self.rm_background(dataset, tg, preds[prev_nbf:nbf + prev_nbf])
            fname = dataset.get_path(iseq)
            hdf = h5py.File(self.dirpath.joinpath(fname.name), 'w')
            self.add_voxinfo(fname, hdf)
            # Also save input/target since they're cropped
            hin, htg = hdf.create_group("/Input"), hdf.create_group("/Target")
            hpred = hdf.create_group("/Prediction")
            for f in range(nbf):
                # data is given as (B, C, W, H, D) and B = 1
                hin.create_dataset(f"vol{f + 1:02d}",
                                   data=inp[f].squeeze().numpy())
                ftg = tg[f].squeeze().numpy()
                fpr = pred[f].squeeze().numpy()
                if fpr.ndim == 4: # Multi class setting
                    hpred.create_dataset(f"anterior-{f + 1:02d}", data=fpr[0])
                    hpred.create_dataset(f"posterior-{f + 1:02d}", data=fpr[1])
                    htg.create_dataset(f"anterior-{f + 1:02d}", data=ftg[0])
                    htg.create_dataset(f"posterior-{f + 1:02d}", data=ftg[1])
                else:
                    hpred.create_dataset(f"vol{f + 1:02d}", data=fpr)
                    htg.create_dataset(f"vol{f + 1:02d}", data=ftg)
            prev_nbf += nbf
            hdf.close()

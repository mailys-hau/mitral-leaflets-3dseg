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
        return save_dir.resolve().expanduser()


class SavePredictedSequence(EnhancedCallback):
    def __init__(self, dirpath="~/Documents/outputs/3dmv-segmentation"):
        #FIXME: Match wandb exp name
        self.dirpath = dirpath

    def on_predict_epoch_end(self, trainer, pl_module, outputs):
        #FIXME? Handle several dataloader
        #FIXME? Handle other than HDFDataset
        #FIXME: Save voxel grid detail on option
        #FIXME? Save alongside input??
        #FIXME: Multiprocess + nice tqdm bar
        all_frames = outputs[0]
        dataset = trainer.predict_dataloaders[0].dataset
        prev_nb_frames = 0
        for i, fname in enumerate(dataset.fnames):
            hdf = h5py.File(self.dirpath.joinpath(fname), 'w')
            res = hdf.create_group("/Predictions")
            nb_frames = dataset.cumulative_frame_len[i + 1] \
                         - dataset.cumulative_frame_len[i]
            for k in range(nb_frames):
                out = all_frames[k + prev_nb_frames].squeeze().numpy()
                res.create_dataset(f"vol{k + 1:02d}", data=out)
            prev_nb_frames = nb_frames
            hdf.close()

    def setup(self, trainer, pl_module, stage):
        self.resolve_dirpath(trainer, "predictions")

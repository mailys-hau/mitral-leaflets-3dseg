import inspect as ispc
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torchmetrics



class EnhancedLightningModule(pl.LightningModule):
    def __init__(self, loss=nn.MSELoss(), optimizer={"name": "Adam", "params": {}},
                 metrics=[], **kwargs):
        # Accept kwargs to allow multi-inheritance
        super(EnhancedLightningModule, self).__init__()
        self.loss = loss
        self.optimizer_config = optimizer
        self._init_metrics(metrics)

    def _init_metrics(self, metrics):
        all_metrics = dict(ispc.getmembers(torchmetrics, ispc.isclass))
        def build_metric(name, *args, **kwargs):
            return all_metrics[name](*args, **kwargs)
        self.train_metrics = nn.ModuleDict({
            m["display_name"]: build_metric(m["name"], *m.get("args", []), **m.get("kwargs", {}))
            for m in metrics })
        self.val_metrics = nn.ModuleDict({
            f"v_{m['display_name']}": build_metric(m["name"], *m.get("args", []), **m.get("kwargs", {}))
            for m in metrics })
        self.test_metrics = nn.ModuleDict({
            m["display_name"]: build_metric(m["name"], *m.get("args", []), **m.get("kwargs", {}))
            for m in metrics })


    def get_progress_bar_dict(self):
        # Don't show version number on progressbar
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


    def _step(self, batch, batch_idx):
        x, y = batch
        #FIXME: Patch for multi-inheritance
        self.preds = self.forward_right(x)
        errs = self.loss(self.preds, y)
        return errs


    def training_step(self, batch, batch_idx):
        _, y = batch
        errs = self._step(batch, batch_idx)
        # Update metrics
        for m in self.train_metrics.values():
            m(self.preds, y)
        self.log_dict(self.train_metrics, on_epoch=True)
        if isinstance(errs, dict): # Used in VAE for example
            self.log_dict(errs, prog_bar=True, on_step=True, on_epoch=True)
            return errs["loss"]
        self.log("loss", errs, prog_bar=True, on_step=True, on_epoch=True)
        return errs

    def validation_step(self, batch, batch_idx):
        _, y = batch
        errs = self._step(batch, batch_idx)
        # Update metrics
        for m in self.val_metrics.values():
            m(self.preds, y)
        self.log_dict(self.val_metrics, on_epoch=True)
        if isinstance(errs, dict): # Used in VAE for example
            errs = {f"v_{k}": v for k, v in errs.items()}
            self.log_dict(errs, prog_bar=True, on_epoch=True)
            return errs["v_loss"]
        self.log("v_loss", errs, prog_bar=True, on_epoch=True)
        return {"v_loss": errs}

    def test_step(self, batch, batch_idx):
        _, y = batch
        errs = self._step(batch, batch_idx)
        # Update metrics
        for m in self.test_metrics.values():
            m(self.preds, y)
        self.log_dict(self.test_metrics, on_epoch=True)
        if isinstance(errs, dict): # Used in VAE for example
            self.log_dict(errs, prog_bar=True, on_epoch=True)
            return errs["loss"]
        self.log("loss", errs, prog_bar=True, on_epoch=True)
        return errs

    def predict(self, batch, batch_idx, dataloader_idx=None):
        errs = self._step(batch, batch_idx)
        self.log_dict(errs)
        return self.preds


    def configure_optimizers(self):
        optimizers = dict(ispc.getmembers(optim, ispc.isclass))
        return optimizers[self.optimizer_config.pop("name")](self.parameters(),
                                                             **self.optimizer_config)

    #def configure_callbacks(self):
    #    return pl.callbacks.ModelCheckpoint(monitor="v_loss")

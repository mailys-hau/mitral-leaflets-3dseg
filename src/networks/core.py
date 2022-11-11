import inspect as ispc
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics



class EnhancedLightningModule(pl.LightningModule):
    def __init__(self, loss=nn.MSELoss(), optimizer={"name": "Adam", "params": {}},
                 metrics=[]):
        super(EnhancedLightningModule, self).__init__()
        self.loss = loss
        self.optimizer_config = optimizer
        self._init_metrics(metrics)

    def _init_metrics(self, metrics):
        all_metrics = dict(ispc.getmembers(torchmetrics, ispc.isclass))
        # We use `nn.ModuleDict` to always be on proper device and such
        self.metrics = nn.ModuleDict()
        def build_metric(name, *args, **kwargs):
            return all_metrics[name](*args, **kwargs)
        self.metrics["mtrain"] = nn.ModuleDict({
            m["display_name"]: build_metric(m["name"],
                                            *m.get("args", []),
                                            **m.get("kwargs", {}))
                for m in metrics })
        self.metrics["mval"] = nn.ModuleDict({
            f"v_{m['display_name']}": build_metric(m["name"],
                                                   *m.get("args", []),
                                                   **m.get("kwargs", {}))
                for m in metrics })
        self.metrics["mtest"] = nn.ModuleDict({
            m["display_name"]: build_metric(m["name"],
                                            *m.get("args", []),
                                            **m.get("kwargs", {}))
                for m in metrics })


    def _step(self, batch, batch_idx):
        x, y = batch
        #FIXME: Patch for multi-inheritance
        self.preds = self.forward(x)
        errs = self.loss(self.preds, y)
        return errs

    def _update_metrics(self, y, mode="train", log=True):
        # Not sure why but key "train" isn't allowed for an `nn.ModuleDict`
        metrics = self.metrics[f"m{mode}"]
        # For classification, most metrics only accept logit target
        int_y = y.to(torch.long)
        for m in metrics.values():
            try:
                m(self.preds, int_y)
            except ValueError:
                m(self.preds, y) # Rare cases where float is needed
        #FIXME: Load by steps for test_step
        self.log_dict(metrics, on_epoch=True)

    def _log_errs(self, errs, name="loss", on_step=False):
        if isinstance(errs, dict): # Used in VAE for example
            if 'v' in name:
                errs = {f"v_{k}": v for k, v in errs.items()}
            self.log_dict(errs, prog_bar=True, on_step=on_step, on_epoch=True)
            return errs[name]
        self.log(name, errs, prog_bar=True, on_step=on_step, on_epoch=True)
        return errs if name == "loss" else {"v_loss": errs}


    def training_step(self, batch, batch_idx):
        _, y = batch
        errs = self._step(batch, batch_idx)
        self._update_metrics(y)
        return self._log_errs(errs, on_step=True)

    def validation_step(self, batch, batch_idx):
        _, y = batch
        errs = self._step(batch, batch_idx)
        self._update_metrics(y, "val")
        return self._log_errs(errs, name="v_loss")

    def test_step(self, batch, batch_idx):
        _, y = batch
        errs = self._step(batch, batch_idx)
        self._update_metrics(y, "test")
        return self._log_errs(errs)

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

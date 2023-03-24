import inspect as ispc
import monai.transforms as mtr
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics

from data import POSTPROCESS
from metrics import MONAI_METRICS
from utils import LinearCosineLR



class EnhancedLightningModule(pl.LightningModule):
    def __init__(self, loss=nn.MSELoss(),
                 optimizer={"name": "Adam", "params": {}}, lr_scheduler=True,
                 final_activation=nn.Softmax(dim=1), postprocess=None,
                 metrics=[]):
        super(EnhancedLightningModule, self).__init__()
        self.loss = loss
        self.optimizer_config = optimizer
        self.lr_scheduler = lr_scheduler
        self.final_activation = final_activation
        self.postprocess = POSTPROCESS.get(postprocess, None)
        self._init_metrics(metrics)
        self.final_activation = final_activation


    def _init_metrics(self, metrics):
        all_metrics = dict(ispc.getmembers(torchmetrics, ispc.isclass))
        all_metrics.update(MONAI_METRICS)
        # We use `nn.ModuleDict` to always be on proper device and such
        self.metrics = nn.ModuleDict({"mtrain": nn.ModuleDict(),
                                      "mval": nn.ModuleDict(),
                                      "mtest": nn.ModuleDict()})
        if self.postprocess:
            self.metrics.update({"pmval": nn.ModuleDict(),
                                 "pmtest": nn.ModuleDict()})
        def build_metric(name, *args, **kwargs):
            return all_metrics[name](*args, **kwargs)
        for m in metrics:
            for mode in self.metrics.keys():
                if mode == "mtrain" and m["name"] in MONAI_METRICS.keys():
                    # Monai computes with `np.array` so use with parsimony
                    continue
                metric = build_metric(m["name"], *m.get("args", []), **m.get("kwargs", {}))
                v = 'v' if "val" in mode else ''
                p = 'p' if 'p' in mode else '' # Postprocess
                display_name = f"{p}{v}_{m['display_name']}".strip('_')
                self.metrics[mode].update({display_name: metric})


    def _step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        # Softmax is baked in `nn.CrossEntropyLoss`, only do it for preds
        sout = self.final_activation(out)
        return sout, self.loss(out, y)

    def _step_end(self, outs, name="loss", mode="train"):
        outs[name] = outs[name].mean()
        if mode != "train" and self.postprocess:
            # Shape is (B, C, W, H, D)
            ppreds = []
            for elt in outs["preds"]: # Apply element wise in the batch
                ppreds.append(self.postprocess(elt))
            outs["ppreds"] = torch.stack(ppreds)
        self._update_metrics(outs, mode)
        return outs


    def _update_metrics(self, outs, mode="train"):
        # "train" key isn't allowed for an `nn.ModuleDict`
        preds, y = outs["preds"], outs["target"]
        metrics = self.metrics[f"m{mode}"]
        on_step = True if mode == "test" else False
        on_epoch = False if mode == "test" else True
        # Distances are not computed for background, we need to set the indexes right
        is_dist = lambda k: "hdf" in k or "masd" in k
        def do_update(k, m):
            if is_dist(k):
                val = m(preds, y)
                if val.shape == ():
                    # Counter PyTorch automatic squeeze of scalars
                    val = val.unsqueeze(0)
            else: # Accuracies prefer booleans as target
                val = m(preds, y.to(torch.bool))
            if val.shape == ():
                self.log_dict({k: val}, on_epoch=on_epoch, on_step=on_step, sync_dist=True)
                return
            self.log_dict({f"{k}/{i + is_dist(k)}": val[i] for i in range(len(val))},
                           on_epoch=on_epoch, on_step=on_step, sync_dist=True)
        for k, m in metrics.items():
            do_update(k, m)
        if "ppreds" in outs.keys():
            # Re-compute for post-processed predictions
            metrics, preds = self.metrics[f"pm{mode}"], outs["ppreds"]
            for k, m in metrics.items():
                do_update(k, m)


    def _log_errs(self, errs, name="loss", on_step=False, on_epoch=True):
        if isinstance(errs, dict): # Used in VAE for example
            if 'v' in name:
                errs = {f"v_{k}": v for k, v in errs.items()}
            self.log_dict(errs, prog_bar=True, on_step=on_step, on_epoch=on_epoch)
            return errs[name]
        self.log(name, errs, prog_bar=True, on_step=on_step, on_epoch=on_epoch,
                 sync_dist=True)
        return {name: errs}


    def training_step_end(self, outs):
        self._step_end(outs)

    def validation_step_end(self, outs):
        self._step_end(outs, "v_loss", "val")

    def test_step_end(self, outs):
        self._step_end(outs, mode="test")

    def predict_step_end(self, outs):
        # Metrics are for test, we're not logging anything here
        return outs # outs = preds in this case

    def training_step(self, batch, batch_idx):
        _, y = batch
        preds, errs = self._step(batch, batch_idx)
        outs = self._log_errs(errs, on_step=True)
        outs.update({"preds": preds, "target": y})
        return outs

    def validation_step(self, batch, batch_idx):
        _, y = batch
        preds, errs = self._step(batch, batch_idx)
        outs = self._log_errs(errs, name="v_loss")
        outs.update({"preds": preds, "target": y})
        return outs

    def test_step(self, batch, batch_idx):
        _, y = batch
        preds, errs = self._step(batch, batch_idx)
        outs = self._log_errs(errs, on_step=True, on_epoch=False)
        outs.update({"preds": preds, "target": y})
        return outs

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        _, y = batch
        preds, _ = self._step(batch, batch_idx)
        if self.postprocess:
            # Shape is (B, C, W, H, D)
            for i in range(len(preds)): # Apply element wise in the batch
                preds[i] = self.postprocess(preds[i])
        return preds


    def configure_optimizers(self):
        # All torch optimizer
        optims = dict(ispc.getmembers(optim, ispc.isclass))
        opt = optims[self.optimizer_config.pop("name")](self.parameters(),
                                                        **self.optimizer_config)

        if self.lr_scheduler:
            nb_batches = len(self.trainer._data_connector._train_dataloader_source.dataloader())
            tot_steps = self.trainer.max_epochs * nb_batches
            lr = opt.defaults["lr"]
            lrs = LinearCosineLR(opt, lr, 100, tot_steps)
        return [opt], [{"scheduler": lrs, "interval": "step"}]

    #def configure_callbacks(self):
    #    return pl.callbacks.ModelCheckpoint(monitor="v_loss")

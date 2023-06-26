import inspect as ispc
import monai.transforms as mtr
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics

from data.postprocess import grey_morphology, MORPHOLOGIES
from metrics import MONAI_METRICS
from utils import LinearCosineLR, TensorList



class EnhancedLightningModule(pl.LightningModule):
    def __init__(self, loss=nn.MSELoss(),
                 optimizer={"name": "Adam"}, lr_scheduler=True,
                 final_activation=nn.Softmax(dim=1), postprocess=None,
                 metrics=[]):
        super(EnhancedLightningModule, self).__init__()
        self.loss = loss
        self.optimizer_config = optimizer
        self.lr_scheduler = lr_scheduler
        self.final_activation = final_activation
        self._init_postprocess(postprocess)
        self._init_metrics(metrics)


    def _init_postprocess(self, postprocess):
        # NB: The order of the given function will be the one used to apply the post process
        #FIXME: Allow to specify transform arguments
        if postprocess is None:
            self.postprocess = None
            return
        monai_transforms = dict(ispc.getmembers(mtr, ispc.isclass))
        if not isinstance(postprocess, list):
            postprocess = [postprocess]
        self.postprocess = []
        for p in postprocess:
            if p in MORPHOLOGIES.keys():
                morpho = lambda vol, name=p: grey_morphology(vol, name)
                self.postprocess.append(morpho)
            elif p in monai_transforms.keys():
                trans = monai_transforms[p]()
                self.postprocess.append(trans)
            else:
                raise ValueError(f"{p} is neither a valid morpholy or monai transform.")

    def _init_metrics(self, metrics):
        all_metrics = dict(ispc.getmembers(torchmetrics, ispc.isclass))
        all_metrics.update(MONAI_METRICS)
        # We use `nn.ModuleDict` to always be on proper device and such
        self.metrics = nn.ModuleDict({"mtrain": nn.ModuleDict(),
                                      "mval": nn.ModuleDict(),
                                      "mtest": nn.ModuleDict()})
        if self.postprocess is not None:
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

    def do_postprocess(self, batch):
        ppreds = []
        for elt in batch: # Apply element wise in the batch
            processed = elt
            for p in self.postprocess: # Post processed function are queued
                processed = p(processed)
            ppreds.append(processed)
        return ppreds

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
            outs["ppreds"] = torch.stack(self.do_postprocess(outs["preds"]))
        self._update_metrics(outs, mode)
        return outs


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
        if self.postprocess is not None:
            # Shape is (B, C, W, H, D)
            preds = self.do_postprocess(preds)
        return preds


    def training_step_end(self, outs):
        self._step_end(outs)

    def validation_step_end(self, outs):
        self._step_end(outs, "v_loss", "val")

    def test_step_end(self, outs):
        self._step_end(outs, mode="test")

    def predict_step_end(self, outs):
        # Metrics are for test, we're not logging anything here
        return outs # outs = preds in this case


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


class ListOutputModule(EnhancedLightningModule):
    """ Extend `EnhancedLightningModule` to handle list of outputs from the network """
    def __init__(self, out_channels=2, loss=nn.MSELoss(),
                 optimizer={"name": "Adam"}, lr_scheduler=True,
                 final_activation=nn.Softmax(dim=1), postprocess=None,
                 metrics=[]):
        self.out_channels = out_channels
        super(ListOutputModule, self).__init__(
                loss=loss, optimizer=optimizer, lr_scheduler=lr_scheduler,
                final_activation=final_activation,
                postprocess=postprocess, metrics=metrics
                )

    def _init_metrics(self, metrics):
        # Metrics are automatically duplicated per out_channels instead of having
        # to specify them all in configuration file
        all_metrics = dict(ispc.getmembers(torchmetrics, ispc.isclass))
        all_metrics.update(MONAI_METRICS)
        # We use `nn.ModuleDict` to always be on proper device and such
        self.metrics = nn.ModuleDict({"mtrain": nn.ModuleDict(),
                                      "mval": nn.ModuleDict(),
                                      "mtest": nn.ModuleDict()})
        if self.postprocess is not None:
            self.metrics.update({"pmval": nn.ModuleDict(),
                                 "pmtest": nn.ModuleDict()})
        def build_metric(name, *args, **kwargs):
            return all_metrics[name](*args, **kwargs)
        for m in metrics:
            for mode in self.metrics.keys():
                if mode == "mtrain" and m["name"] in MONAI_METRICS.keys():
                    # Monai computes with `np.array` so use with parsimony
                    continue
                for i in range(self.out_channels):
                    metric = build_metric(m["name"], *m.get("args", []), **m.get("kwargs", {}))
                    v = 'v' if "val" in mode else ''
                    p = 'p' if 'p' in mode else '' # Postprocess
                    display_name = f"{p}{v}_{m['display_name']}/{i + 1}".strip('_')
                    self.metrics[mode].update({display_name: metric})

    # *_step and *_step_end are the same as mother class
    def _step(self, batch, batch_idx):
        x, ys = batch
        outs = self.forward(x) # Return `nn.TensorList`
        # Softmax is baked in `nn.CrossEntropy`, only do it for preds
        souts = TensorList(*map(self.final_activation, outs))
        # Stack on batch dim
        #FIXME: For unknown reason, ys is nested of one level here
        return souts, self.loss(torch.cat(outs, dim=0), torch.cat(ys[0], dim=0))

    def _step_end(self, outs, name="loss", mode="train"):
        outs[name] = outs[name].mean()
        if mode != "train" and self.postprocess:
            fmap = lambda x: torch.stack(self.do_postprocess(x))
            outs["ppreds"] = TensorList(*map(fmap, outs["preds"]))
        self._update_metrics(outs, mode)
        return outs

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        _, y = batch
        preds, _ = self._step(batch, batch_idx)
        if self.postprocess is not None:
            # Shape is (B, C, W, H, D)
            fmap = lambda x: torch.stack(self.do_postprocess(x))
            preds = TensorList(*map(fmap, preds))
        return preds


    def _update_metrics(self, outs, mode="train"):
        # "train" key isn't allowed for an `nn.ModuleDict`
        #FIXME: For unknown reason, ys is nested of one level here
        preds, yy = outs["preds"], outs["target"][0]
        metrics = self.metrics[f"m{mode}"]
        on_step = True if mode == "test" else False
        on_epoch = False if mode == "test" else True
        def do_update(k, m, i):
            try:
                val = m(preds[i], yy[i])
            except TypeError: # Accuracies prefer booleans as target
                val = m(preds[i], yy[i].to(torch.bool))
            self.log_dict({f"{k}": val}, on_epoch=on_epoch, on_step=on_step, sync_dist=True)
        for k, m in metrics.items():
            i = int(k.split('/')[-1]) - 1
            do_update(k, m, i)
        if "ppreds" in outs.keys():
            # Re-compute for post-processed predictions
            metrics, preds = self.metrics[f"pm{mode}"], outs["ppreds"]
            for k, m in metrics.items():
                i = int(k.split('/')[-1]) - 1
                do_update(k, m, i)

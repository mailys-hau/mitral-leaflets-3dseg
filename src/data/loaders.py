import torch

from torch.utils.data import DataLoader

from data.datasets import DummyDataset, HDFDataset


_datasets = {"HDFDataset": HDFDataset}



def load_data(name, test=False, debug=False, **kwargs):
    kwdataset = kwargs.pop("dataset")
    dataset = _datasets[name]
    prefix = kwdataset.pop("prefix")
    files = kwdataset.pop("files")
    if test:
        if debug:
            testset = DummyDataset(3)
        else:
            testset = dataset(prefix, files["test"]["files"],
                              files["test"]["total_frames"], **kwdataset)
        return DataLoader(testset, **kwargs)
    else:
        bs = kwargs.pop("batch_size", 16)
        if debug:
            trainset, valset = DummyDataset(10), DummyDataset(3)
        else:
            trainset = dataset(prefix, files["train"]["files"],
                               files["train"]["total_frames"], **kwdataset)
            valset = dataset(prefix, files["validation"]["files"],
                             files["validation"]["total_frames"], **kwdataset)
        trainloader = DataLoader(trainset, batch_size=bs, shuffle=True, **kwargs)
        valloader = DataLoader(valset, batch_size=bs, **kwargs)
        return trainloader, valloader

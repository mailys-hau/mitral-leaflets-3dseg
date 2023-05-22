import torch

from torch.utils.data import DataLoader

from data.collates import *
from data.datasets import *


_datasets = {"DummyDataset": DummyDataset,
             "FrameDataset": FrameDataset,
             "MiddleFrameDataset": MiddleFrameDataset,
             "ListMiddleFrameDataset": ListMiddleFrameDataset,
             "SequenceDataset": SequenceDataset}

_collates = {"collate_tensorlist": collate_tensorlist}



def load_data(name, test=False, **kwargs):
    kwdataset = kwargs.pop("dataset")
    collate_fn = _collates.get(kwargs.pop("collate_fn", None), None)
    if name == "DummyDataset": # Debug case
        nb_dummies = kwdataset.pop("nb_dummies", 10)
        if test:
            testset = DummyDataset(3, **kwdataset)
            return DataLoader(testset, **kwargs)
        else:
            trainset = DummyDataset(nb_dummies, **kwdataset)
            valset = DummyDataset(3, **kwdataset)
    else: # Usual case
        dataset = _datasets[name]
        prefix, files = kwdataset.pop("prefix"), kwdataset.pop("files")
        if test:
            # Fix non-random values outside training
            kwdataset["resize"], kwdataset["augmentation"] = "center", False
            testset = dataset(prefix, files["test"]["files"], **kwdataset)
            return DataLoader(testset, **kwargs, collate_fn=collate_fn)
        else:
            trainset = dataset(prefix, files["train"]["files"], **kwdataset)
            # Fix non-random values outside training
            kwdataset["resize"], kwdataset["augmentation"] = "center", False
            valset = dataset(prefix, files["validation"]["files"], **kwdataset)
    trainloader = DataLoader(trainset, shuffle=True, **kwargs, collate_fn=collate_fn)
    valloader = DataLoader(valset, **kwargs, collate_fn=collate_fn)
    return trainloader, valloader

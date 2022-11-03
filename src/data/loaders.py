import torch

from torch.utils.data import DataLoader

from data.datasets import HDFDataset


_datasets = {"HDFDataset": HDFDataset}



def load_data(name, test=False, **kwargs):
    kwdataset = kwargs.pop("dataset")
    dataset = _datasets[name]
    if test:
        testset = dataset(kwdataset["prefix"], kwdataset["files"]["test"]["files"],
                           kwdataset["files"]["test"]["total_frames"])
        return DataLoader(testset, **kwargs)
    else:
        trainset = dataset(kwdataset["prefix"], kwdataset["files"]["train"]["files"],
                           kwdataset["files"]["train"]["total_frames"])
        valset = dataset(kwdataset["prefix"], kwdataset["files"]["validation"]["files"],
                         kwdataset["files"]["validation"]["total_frames"])
        trainloader = DataLoader(trainset, batch_size=kwargs.pop("batch_size", 16),
                                 shuffle=True, **kwargs)
        valloader = DataLoader(valset, **kwargs)
        return trainloader, valloader

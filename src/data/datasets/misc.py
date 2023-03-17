import torch

from torch.utils.data import Dataset



class DummyDataset(Dataset):
    def __init__(self, nb_dummies, spatial_size=[128, 128, 128],
                 multiclass=False, **kwargs):
        # Accept kwargs in case default config pass arguments from regular dataset
        self.inputs = torch.rand(nb_dummies, *spatial_size).to(torch.float)
        nc = 3 if multiclass else 2
        self.outputs = torch.randint(0, 2, (nb_dummies, nc, *spatial_size)).to(torch.float)
        self.multiclass = multiclass
        self.nb_dummies = nb_dummies

    def __getitem__(self, i):
        return self.inputs[i].unsqueeze(0), self.outputs[i]

    def __len__(self):
        return self.nb_dummies

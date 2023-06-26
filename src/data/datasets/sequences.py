from data.datasets.core import _HDFDataset, _ListHDFDataset



class SequenceDataset(_HDFDataset):
    """ Load all frames of given sequences """
    def __init__(self, data_dir, hdfnames, multiclass=False,
                 resize="center-random", spatial_size=[128, 128, 128],
                 norm="256", contrast=None, augmentation=False, cache=False):
        super(SequenceDataset, self).__init__(
                data_dir, hdfnames, multiclass, resize, spatial_size, norm,
                contrast, augmentation, cache)

    def __getitem__(self, i):
        iseq = self.sequence_indexes[i]
        iframe = i - self.cumulative_nbf[iseq]
        return self.get_volumes(i, iseq, iframe)

    def get_sequence(self, iseq):
        # Get all frames from a specific sequence at once
        inseq, tgseq = [], []
        nbf = self.cumulative_nbf[iseq + 1] - self.cumulative_nbf[iseq]
        for iframe in range(nbf):
            inp, tg = self[iseq + iframe] #?
            inseq.append(inp), tgseq.append(tg)
        return inseq, tgseq

    def __len__(self):
        return len(self.sequence_indexes)


class ListSequenceDataset(_ListHDFDataset):
    def __init__(self, data_dir, hdfnames, resize="center-random",
                 spatial_size=[128, 128, 128], norm="256", contrast=None,
                 augmentation=False, cache=False):
        super(ListSequenceDataset, self).__init__(
                data_dir, hdfnames, resize, spatial_size, norm, contrast,
                augmentation, cache)

    def __getitem__(self, i):
        iseq = self.sequence_indexes[i]
        iframe = i - self.cumulative_nbf[iseq]
        return self.get_volumes(i, iseq, iframe)

    def get_sequence(self, iseq):
        # Get all frames from a specific sequence at once
        inseq, tgseq = [], []
        nbf = self.cumulative_nbf[iseq + 1] - self.cumulative_nbf[iseq]
        for iframe in range(nbf):
            inp, tg = self[iseq + iframe] #?
            inseq.append(inp), tgseq.append(tg)
        return inseq, tgseq

    def __len__(self):
        return len(self.sequence_indexes)

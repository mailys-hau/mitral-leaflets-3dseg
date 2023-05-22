from data.datasets.core import _HDFDataset, _ListHDFDataset



class FrameDataset(_HDFDataset):
    """ Load one specific frame per sequence of given HDF """
    def __init__(self, data_dir, hdfnames, frame_index=0, multiclass=False,
                 resize="center-random", spatial_size=[128, 128, 128],
                 norm="256", contrast=None, augmentation=False, cache=True):
        super(FrameDataset, self).__init__(
                data_dir, hdfnames, multiclass, resize, spatial_size, norm,
                contrast, augmentation, cache)
        self.frame_index = frame_index

    def _get_frame_index(self, iseq):
        if callable(self.frame_index):
            nbf = self.cumulative_nbf[iseq + 1] - self.cumulative_nbf[iseq]
            return self.frame_index(nbf)
        return self.frame_index # Fixed integer

    def get_sequence(self, iseq):
        # Mock sequence format to ease the callback process
        inp, tg = self[iseq]
        return [inp], [tg]

    def __getitem__(self, i):
        # i = iseq in this case
        iframe = self._get_frame_index(i)
        return self.get_volumes(i, i, iframe)

    def __len__(self):
        return len(self.fnames)

class MiddleFrameDataset(FrameDataset):
    """ AutoMVQ's reference frame is the middle one of the sequence """
    def __init__(self, data_dir, hdfnames, multiclass=False,
                 resize="center-random", spatial_size=[128, 128, 128],
                 norm="256", contrast=None, augmentation=False, cache=True):
        super(MiddleFrameDataset, self).__init__(
                data_dir, hdfnames, lambda i: int(i / 2), multiclass, resize,
                spatial_size, norm, contrast, augmentation, cache)


class ListMiddleFrameDataset(_ListHDFDataset):
    """ AutoMVQ's reference frame is the middle one of the sequence """
    def __init__(self, data_dir, hdfnames, resize="center-random", spatial_size=[128, 128, 128],
                 norm="256", contrast=None, augmentation=False, cache=True):
        super(ListMiddleFrameDataset, self).__init__(
                data_dir, hdfnames, resize, spatial_size, norm, contrast,
                augmentation, cache)
        self.frame_index = lambda i: int(i / 2)

    def _get_frame_index(self, iseq):
        if callable(self.frame_index):
            nbf = self.cumulative_nbf[iseq + 1] - self.cumulative_nbf[iseq]
            return self.frame_index(nbf)
        return self.frame_index # Fixed integer

    def get_sequence(self, iseq):
        # Mock sequence format to ease the callback process
        inp, tg = self[iseq]
        return [inp], [tg]

    def __getitem__(self, i):
        # i = iseq in this case
        iframe = self._get_frame_index(i)
        return self.get_volumes(i, i, iframe)

    def __len__(self):
        return len(self.fnames)

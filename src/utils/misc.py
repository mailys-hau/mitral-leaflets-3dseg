import yaml

from collections.abc import Iterable, Mapping
from pathlib import Path



class InclusiveLoader(yaml.SafeLoader):
    """ Allow use of `!include filename.yml` statement in yaml file """
    def __init__(self, stream):
        super(InclusiveLoader, self).__init__(stream)

    def include(self, node):
        fname = Path(self.construct_scalar(node)).expanduser().resolve()
        with open(fname, 'r') as fd:
            return yaml.load(fd, InclusiveLoader)


def rec_update(d, u):
    """ Recursively update dict of any depth """
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = rec_update(d.get(k, {}), v)
        else:
            if k in d.keys() and isinstance(v, Iterable) and not isinstance(v, str):
                d[k].extend(v)
            else:
                d[k] = v
    return d

def rec_flatten(l):
    out = l.__class__() # Agnostic for list or TensorList
    for e in l:
        if isinstance(e, list):
            out.extend(rec_flatten(e))
        else:
            out.append(e)
    return out

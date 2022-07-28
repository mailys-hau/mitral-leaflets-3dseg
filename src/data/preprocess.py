"""
Parse HDF file of TEE to retrieve useful information and split data between
train, validation, and test sets
"""

import click as cli
import h5py
import random as rd
import sys
import yaml

from pathlib import Path

rd.seed(42)




@cli.command(context_settings={"help_option_names": ["-h", "--help"],
                               "show_default": True})
@cli.argument("pdata", type=cli.Path(exists=True, resolve_path=True, file_okay=False,
              path_type=Path))
@cli.option("--train-ratio", "-tr", "rtrain", type=cli.FloatRange(0, 1), default=0.7,
            help="Ratio of the whole data used for training dataset.")
@cli.option("--validation-ratio", "-v", "rval", type=cli.FloatRange(0, 1), default=0.1,
            help="Ratio of the whole data used for validation dataset.")
@cli.option("--test-ratio", "-te", "rtest", type=cli.FloatRange(0, 1), default=0.2,
            help="Ratio of the whole data used for testing dataset.")
@cli.option("--output", "-o", "ofname", show_default=False,
            type=cli.Path(writable=True, path_type=Path), default="data-split.yml",
            help="Where to save split and frames number. [default: PDATA/data-split.yml]")
def build_dataset(pdata, rtrain, rval, rtest, ofname):
    """
    Retrieve number of frames from each sequence (i.e. one HDF file) and split the
    whole data to train, validation, and test sets.\n
    /!\ Split is performe accross sequences and not frames. /!\ 

    PDATA    DIR     Path to directory containing data in HDF format.
    """
    assert (rtrain + rval + rtest) == 1, \
           "Train, validation and test ratios must sum to 1."
    out = {}
    data = []
    ofname = pdata.joinpath(ofname) if ("-o" or "--output" in sys.argv[1:]) \
                                    else ofname.resolve()
    for fname in pdata.iterdir():
        if fname.suffix != ".h5":
            print("Ignoring {fname.name}, not an HDF.")
            continue
        hdf = h5py.File(fname, 'r')
        data.append((fname.name, int(hdf["ImageGeometry"]["frameNumber"][()])))
        hdf.close()
    # Split into train/validation/test sets
    rd.shuffle(data)
    nb = len(data)
    tridx, validx, teidx = int(rtrain * nb), int(rtrain * nb), int(rtrain * nb)
    train, val, test = data[:tridx], data[tridx:validx], data[validx:]
    out = {"train": {"files": train, "total_frames": sum([t[1] for t in train])},
           "validation": {"files": val, "total_frames": sum([t[1] for t in val])},
           "test": {"files": test, "total_frames": sum([t[1] for t in test])}}
    out["total_frames"] = sum([subout["total_frames"] for subout in out.values()])
    with open(ofname, 'w') as fd:
        #TODO? Order keys
        yaml.dump(out, fd, default_flow_style=False)
    print(f"Split file available at {ofname}")



if __name__ == "__main__":
    build_dataset()

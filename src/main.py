import click as cli
import yaml

from copy import deepcopy
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything

from data import load_data
from networks import build_model
from utils import InclusiveLoader, rec_update




seed_everything(404)



@cli.group(context_settings={"help_option_names": ["-h", "--help"]})
@cli.option("-c", "--config-file", type=cli.Path(exists=True), default=None,
            help="YML to configure called command.")
@cli.option("--debug", is_flag=True,
            help="Put results in Debug DB of WandBFlow.")
@cli.pass_context
def main(ctx, config_file, debug):
    with open(f"../config/default-{ctx.invoked_subcommand}.yml", 'r') as fd:
        full_config = yaml.load(fd, InclusiveLoader)
    if config_file:
        with open(config_file, 'r') as ffd:
            config = yaml.load(ffd, InclusiveLoader)
        full_config = rec_update(full_config, config)
    group = "debug" if debug else config["network"]["name"]
    ctx.obj = {"config": full_config, "group": group}


@main.command(name="train", short_help="Training.")
@cli.pass_context
def train(ctx): # FIX
    """ The eye of the tiger """
    config = deepcopy(ctx.obj["config"])
    print("Loading data...")
    cdata = config["data"]
    trloader, valoader = load_data(cdata["dataset"].pop("name"), **cdata)
    print("Building network...")
    cnet = config["network"]
    net = build_model(cnet.pop("name"), cnet.pop("loss"), cnet.pop("optimizer"), **cnet)
    logdir = Path("../outputs").resolve()
    logdir.mkdir(exist_ok=True) # Create if non-existent
    #TODO: Add tag to logger
    wandblog = WandbLogger(project="3D-MA-segmentation", group=ctx.obj["group"],
                           job_type="train", entity="mailys-hau", save_dir=str(logdir))
    trainer = Trainer(callbacks=[ModelCheckpoint(monitor="v_loss")], logger=wandblog,
                      **config["trainer"])
    print("Training...")
    trainer.fit(net, trloader, valoader)
    # Save full training config
    trainer.logger.log_hyperparams(ctx.obj["config"])

@main.command(name="test", short_help="Testing.")
@cli.pass_context
def test(ctx):
    """ Let's see if your network work """
    config = deepcopy(ctx.obj["config"])
    print("Loading data...")
    teloader = load_data(cdata["dataset"].pop("name"), test=True, **cdata)
    print("Building network...")
    if not "weights" in config["network"]:
        raise UserWarning("No weights were given for the network. Results will be random.")
    logdir = Path("../outputs").resolve()
    logdir.mkdir(exist_ok=True) # Create if non-existent
    #TODO: Add tag to logger
    wandblog = WandbLogger(project="3D-MA-segmentation", group=ctx.obj["group"],
                           job_type="eval", entity="mailys-hau", save_dir="../outputs")
    tester = Trainer(logger=wandblog, **config["tester"])
    tester.test(net, teloader)
    # Save full training config
    tester.logger.log_hyperparams(ctx.obj["config"])



if __name__ == "__main__":
    main()

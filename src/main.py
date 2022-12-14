import click as cli
import yaml

from copy import deepcopy
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything

from callbacks import *
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
    group = "debug" if debug else full_config["network"]["name"]
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
    # TODO? Match loggdir with checkpoint dir
    logdir = Path("../outputs").resolve()
    logdir.mkdir(exist_ok=True) # Create if non-existent
    #TODO: Add tag to logger
    wandblog = WandbLogger(project="3DMV-segmentation", group=ctx.obj["group"],
                           job_type="train", entity="tee-4d", save_dir=str(logdir))
    # Save full training config (before training in case of crash)
    wandblog.experiment.config.update(ctx.obj["config"])
    trainer = Trainer(callbacks=[EnhancedModelCheckpoint(monitor="v_loss")],
                      logger=wandblog, **config["trainer"])
    print("Training...")
    trainer.fit(net, trloader, valoader)


@main.command(name="test", short_help="Testing.")
@cli.option("--eval/--no-eval", "eval_net", is_flag=True, default=True,
            help="Do an evaluation run of the given network.")
@cli.option("--predict/--no-predict", is_flag=True, default=False,
            help="Predict on given dataset with given network.")
@cli.pass_context
def test(ctx, eval_net, predict):
    """ Let's see if your network work """
    config = deepcopy(ctx.obj["config"])
    print("Loading data...")
    cdata = config["data"]
    teloader = load_data(cdata["dataset"].pop("name"), test=True, **cdata)
    print("Building network...")
    cnet = config["network"]
    if not "weights" in cnet:
        raise UserWarning("No weights were given for the network. Results will be random.")
    net = build_model(cnet.pop("name"), cnet.pop("loss"), cnet.pop("optimizer"), **cnet)
    logdir = Path("../outputs").resolve()
    logdir.mkdir(exist_ok=True) # Create if non-existent
    #TODO: Add tag to logger
    wandblog = WandbLogger(project="3DMV-segmentation", group=ctx.obj["group"],
                           job_type="eval", entity="tee-4d", save_dir="../outputs")
    # Save full testing config (before testing in case of crash)
    wandblog.experiment.config.update(ctx.obj["config"])
    tester = Trainer(logger=wandblog, **config["tester"],
                     max_epochs=-1, # Remove warning
                     callbacks=[SavePredictedSequence(), Plot4D()])
    #FIXME: Do the same thing twice, inneficient
    if eval_net:
        tester.test(net, teloader)
    if predict:
        foo = tester.predict(net, teloader)



if __name__ == "__main__":
    main()

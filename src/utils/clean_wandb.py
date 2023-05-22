import click as cli
import wandb

from pathlib import Path
from shutil import move, rmtree



@cli.command(context_settings={"help_option_names": ["-h", "--help"],
                               "show_default": True})
@cli.argument("path", type=cli.Path(exists=True, file_okay=False, writable=True,
                                    resolve_path=True, path_type=Path))
@cli.option("--entity", "-e", type=str, default="tee-4d",
            help="WandB's entity name. Can be your username.")
@cli.option("--project", "-p", type=str, default=None,
            help="WandB's project name where you want to compare")
@cli.option("--group", "-g", type=str, default=None,
            help="Only look at a specific group.")
def clean_runs(path, entity, project, group):
    """
    Delete files associated to deleted runs in WandB.

    PATH    DIR    Directory to clean.
    """
    project = path.stem if project is None else project
    api = wandb.Api(overrides={"entity": entity, "project": project})
    for exp in path.iterdir():
        run_id = exp.stem.split('_')[-1]
        try:
            #FIXME: if group given, filter
            run = api.run(f"{project}/{run_id}")
        except wandb.errors.CommError as err: # Run not found
            #print(err)
            print(f"Deleting {exp.stem}")
            rmtree(exp)



if __name__ == "__main__":
    clean_runs()

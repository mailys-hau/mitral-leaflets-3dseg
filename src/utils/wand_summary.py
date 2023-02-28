import click as cli
import wandb

from itertools import accumulate



ALL_METRICS = {True: ["hdf95/1", "hdf95/2", "masd/1", "masd/2", "dice",
                      "acc/1", "acc/2", "prec/1", "prec/2", "rec/1", "rec/2"],
               False: ["hdf95/1", "masd/1", "dice", "acc/1", "prec/1", "rec/1"]}
CORE_METRICS = {True: ["hdf95/1", "hdf95/2", "masd/1", "masd/2", "dice"],
                False: ["hdf95/1", "masd/1", "dice"]}
# Hardcoded for lazyness
NBF = [7, 15, 14, 10, 5, 14, 12, 12, 3, 11, 5, 77]
CUMNBF = list(accumulate(NBF, initial=0))



@cli.command(context_settings={"help_option_names": ["-h", "--help"],
                               "show_default": True})
@cli.argument("run_id", type=str)
@cli.option("--accuracies/--core", "-a/-C", default=True,
            help="Also compute summary for accuracies.")
@cli.option("--multiclass/--binary", "-m/-b", "multi", default=False,
            help="Whether your run was binary or multi segmentation.")
@cli.option("--per-sequence", "-s / -S", is_flag=True, default=True,
            help="Also compute a summary per sequence.")
def summarize(run_id, accuracies, multi, per_sequence):
    """ Pouet pouet pouet """
    api = wandb.Api()
    run = api.run(f"tee-4d/3DMV-segmentation/{run_id}")
    metrics = ALL_METRICS[multi] if accuracies else CORE_METRICS[multi]
    res = run.history(keys=metrics) # Return `pandas.DataFrame`
    res.drop("_step", axis=1, inplace=True)
    res = res.astype(float) # Just in case
    res[res.columns[res.columns.str.contains("masd|hdf")]] *= 0.7 # Vox to mm
    res[res.columns[~res.columns.str.contains("masd|hdf")]] *= 100 # Accuracies
    print("Gobal summary: ")
    for c in res.columns:
        print(f"  {c.capitalize():<{7}}: {res[c].mean():>{5}.2f} ± {res[c].std():.2f}")
    if per_sequence:
        print("\nPer sequence summary:")
        for i in range(len(CUMNBF) - 1):
            seq = res.iloc[CUMNBF[i]:CUMNBF[i + 1]]
            print("  ", end='')
            print(" | ".join([
                f"{c.capitalize()}: {seq[c].mean():>{5}.2f} ± {seq[c].std():>{5}.2f}"
                    for c in seq.columns]))



if __name__ == "__main__":
    summarize()

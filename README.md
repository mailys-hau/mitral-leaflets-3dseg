# MV-3DSEG
Deep learning-based segmentation of mitral leaflets on 3D-TEE based on [MONAI framework](https://monai.io) and [PyTorchLightning](https://lightning.ai/)

A `requirements.txt` is available with the list of modules used and their version. The experiments were done using Python3.9.

**NB:** *All experiments are automatically logged to [Weight & Bias](https://wandb.ai/).*

## Usage
To train a network, run `$ python main.py -c <path-to-config.yml> train`. Any option specified in both your configuration file and `config/default-train.yml` will get its value taken from the provided configuration file. An example of configuration file can be found in `config/`, [here](https://github.com/mailys-hau/mitral-leaflets-3dseg/blob/main/config/final_train.yml), as well as the default parameters in both `config/default-train.yml` and `default-test.yml`.

### Data format
Data is expected to be located in the directory given as a prefix (you can list several directories). A YAML file describing the dataset split should also be included and referenced in the configuration file. Below is an example of the expected data split format:
```
test:
  files:
  - - filename1.h5
    - 2 # Number of systole frames in HDF file
  - - filename2.h5
    - 4
  - - filename3.h5
    - 3
  total_frames: 9
train:
  files:
  - - filename4.h5
    - 2
  - - filename5.h5
    - 3
  - - filename6.h5
    - 5
  - - filename7.h5
    - 1
  total_frames: 11
validation:
  files:
  - - filename8.h5
    - 2
  - - filename9.h5
    - 4
  total_frames: 6
total_frames: 26
```
HDFs follow the same organisation as described [here](https://github.com/mailys-hau/echovox#output).

### Evaluation loops
To evaluate the network on the given metrics, run `$ python main.py -c <path-to-config.yml> test`. To also save the network's predictions, run `$ python main.py -c <path-to-config.yml> test --predict`. This will also generate the plots using PyTorchLightning's callbacks and [echoviz](https://pypi.org/project/echoviz-MALOU/). Predictions are saved in `~/Documents/outputs/<WandB-experiment-name_WandB-experiment-id>/predictions/` using the same filename as the data inputted in the network and following the HDF structure described below:
```
|-- Input/
    |-- vol01
    |-- vol02
    |-- ...
|-- Target/
    |-- anterior-01
    |-- anterior-02
    |-- ...
    |-- posterior-01
    |-- ...
|-- Prediction/
    |-- anterior-01
    |-- anterior-02
    |-- ...
    |-- posterior-01
    |-- ...
|-- VolumeGeometry/
    |-- directions
    |-- origin
    |-- resolution
```

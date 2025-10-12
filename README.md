# IMD-Net

This repo contains the original implementation of [IMD-Net: A Deep Learning-Based Icosahedral Mesh Denoising Network](https://ieeexplore.ieee.org/document/9749069).

## Installation

[uv](https://docs.astral.sh/uv/getting-started/installation/) is used as package manager for this repo. After cloning, enter the repo and run

```shell
uv sync
```

to set up the environment.

## Training and evaluation
For a given dataset at PATH_TO_DATASET create training and validation splits via
```shell
uv run preprocessing/partition.py --data_dir PATH_TO_DATASET --out_dir ./examples --seed 42
```
This will create files containing reproducible training and validation splits of the data. Now, training can be run with:

```shell
uv run imd_net/main.py fit -c configs/abc_small_gwn02.yaml --data.data_dir PATH_TO_DATASET --data.splits_dir ./examples
```

And inference like this:

```shell
mkdir ./predictions
uv run imd_net/main.py test -c configs/abc_small_gwn02.yaml --data.data_dir PATH_TO_DATASET --data.splits_dir ./examples --ckpt_path PATH_TO_CHECKPOINT --trainer.callbacks+=SavePredictionsCallback --trainer.callbacks.save_dir ./predictions
```
The above line also writes the predictions of the val/test set to disk.

## Data

A subset of the data used in the paper can be found here: [abc_small_gwn02](https://drive.google.com/drive/folders/1hfvuREOLooRE6vFd5qKMkyzAXiTYcjpy?usp=sharing)
The data is a random subset of the [ABC dataset](https://deep-geometry.github.io/abc-dataset/) and as the name indicates, gaussian white noise with ~N(0, 0.02) was added to the vertex positions. Each mesh in the dataset is represented by a `.npz` file containing the following entries:
- `data`: Noisy vertex positions
- `targets`: Ground truth vertex positions

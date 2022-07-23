import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pathlib import Path
from argparse import ArgumentParser


class IcoMeshDataset(Dataset):
    """Data loader for IcoMesh datasets."""

    def __init__(self,
                 data_dir: str,
                 partition: str = "train",
                 subdivisions: int = 5,
                 data_internal_scale: float = None,
                 load_data_to_ram: bool = False,
                 vmask_thres: float = None,
                 vmask_weight: float = None,
                 fmask_thres: float = None,
                 fmask_weight: float = None
                 ):
        """
        Args:
            data_dir: path to data directory
            partition: train or test
        """
        dataset_path = Path(data_dir)

        assert (partition in ["train", "test", "val", "all"])
        assert subdivisions >= 0, 'subdivision must be >= 0'
        assert ~dataset_path.exists() or ~dataset_path.is_dir(), \
            '{} does not exist or is not a dir'.format(dataset_path)

        # get file list
        splits_dir = Path(__file__).parent.absolute() / "splits"
        if partition == 'all':
            lines = []
            for part in ["train", "test", "val"]:
                split_file = splits_dir / "{}_split_{}.txt".format(part, dataset_path.name)
                with open(split_file, "r") as f:
                    lines.extend(f.readlines())
        else:
            split_file = splits_dir / "{}_split_{}.txt".format(partition, dataset_path.name)
            with open(split_file, "r") as f:
                lines = f.readlines()
        self.flist = [dataset_path.joinpath(l.replace('\n', '').split(' ')[0]) for l in lines]

        self.icomap_width = 2 ** (subdivisions + 1)
        self.data_internal_scale = data_internal_scale
        self.vmask_thres = vmask_thres
        self.vmask_weight = vmask_weight
        self.fmask_thres = fmask_thres
        self.fmask_weight = fmask_weight

        self.loaded_items = [None] * len(self.flist)
        if load_data_to_ram:
            for idx in range(len(self.flist)):
                self.loaded_items[idx] = self._loaditem(idx)

    def __len__(self):
        return len(self.flist)

    def __getitem__(self, idx):
        return self.loaded_items[idx] or self._loaditem(idx)

    def _loaditem(self, idx):
        fname = self.flist[idx]
        data = np.load(str(fname))["data"][:3, :-2]
        data = data.reshape(data.shape[0], -1, self.icomap_width)

        targets = np.load(str(fname))["targets"]

        if self.data_internal_scale:
            data = data * self.data_internal_scale
            targets[:3, :] = targets[:3, :] * self.data_internal_scale

        # compute vertex mask for enhancing features
        feature_mask = np.ones([targets.shape[1]])
        if self.vmask_thres:
            vertex_mask = np.load(str(fname))["v_max_dangles"].flatten()
            feature_mask[vertex_mask > self.vmask_thres] = self.vmask_weight
        feature_mask = torch.from_numpy(feature_mask).unsqueeze(-1).expand(-1, 3)

        return data, (targets.transpose(), feature_mask), idx


class IcoMeshDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            batch_size: int,
            subdivisions: int,
            data_internal_scale: float,
            load_data_to_ram: bool = False,
            vmask_thres: float = None,
            vmask_weight: float = None,
            fmask_thres: float = None,
            fmask_weight: float = None):
        super().__init__()

        # unpack input
        data_dir = Path(data_dir)

        # check sanity of input
        assert ~data_dir.exists() or ~data_dir.is_dir(), \
            '{} does not exist or is not a dir'.format(data_dir)
        assert subdivisions >= 0, 'subdivision must be >= 0'

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.subdivisions = subdivisions
        self.data_internal_scale = data_internal_scale
        self.load_data_to_ram = load_data_to_ram
        self.num_workers = 0 if self.load_data_to_ram else os.cpu_count()
        self.vmask_thres = vmask_thres
        self.vmask_weight = vmask_weight
        self.fmask_thres = fmask_thres
        self.fmask_weight = fmask_weight

        splits_dir = Path(__file__).parent.absolute() / "splits"

        self.train_file = splits_dir / "train_split_{}.txt".format(Path(self.data_dir).name)
        self.val_file = splits_dir / "val_split_{}.txt".format(Path(self.data_dir).name)
        self.test_file = splits_dir / "test_split_{}.txt".format(Path(self.data_dir).name)

        self.train_count = 0
        self.val_count = 0
        self.test_count = 0

    def _create_val_dataloader(self):
        dataset = IcoMeshDataset(str(self.data_dir), "val", self.subdivisions, self.data_internal_scale,
                                 self.load_data_to_ram, self.vmask_thres, self.vmask_weight, self.fmask_thres,
                                 self.fmask_weight)
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                            drop_last=False, pin_memory=True)
        return loader

    def prepare_data(self):
        assert ~self.train_file.exists() or ~self.train_file.is_file(), \
            '{} does not exist or is not a dir'.format(self.train_file)
        assert ~self.val_file.exists() or ~self.val_file.is_file(), \
            '{} does not exist or is not a dir'.format(self.val_file)
        assert ~self.test_file.exists() or ~self.test_file.is_file(), \
            '{} does not exist or is not a dir'.format(self.test_file)

        with self.train_file.open('r') as f:
            i = -1
            for i, l in enumerate(f):
                pass
            self.train_count = i + 1

        with self.val_file.open('r') as f:
            i = -1
            for i, l in enumerate(f):
                pass
            self.val_count = i + 1

        with self.test_file.open('r') as f:
            i = -1
            for i, l in enumerate(f):
                pass
            self.test_count = i + 1
            # lines = f.readlines()
            # tmp_list = [(self.data_dir / line.strip()) for line in lines]

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        dataset = IcoMeshDataset(str(self.data_dir), "train", self.subdivisions, self.data_internal_scale,
                                 self.load_data_to_ram, self.vmask_thres, self.vmask_weight, self.fmask_thres,
                                 self.fmask_weight)
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                            drop_last=True, pin_memory=True)
        return loader

    def val_dataloader(self):
        return self._create_val_dataloader()

    def test_dataloader(self):
        return None

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--data_dir", default='./', type=str,
                            help='path to the directory that contains the dataset')
        parser.add_argument('-b', '--batch_size', default=48, type=int,
                            help='the batch size')

        parser.add_argument('--load_data_to_ram', action='store_true',
                            help='load the entire dataset to the RAM for training')

        # params for weighting of vertices/faces (based on dihedral angles of connected edges)
        parser.add_argument('--vmask_thres', default=None, type=float,
                            help='the threshold dihedral angle above which vertices get weighted')
        parser.add_argument('--vmask_weight', default=None, type=float,
                            help='the factor to use for weighted vertices')
        parser.add_argument('--fmask_thres', default=None, type=float,
                            help='the threshold dihedral angle above which faces get weighted')
        parser.add_argument('--fmask_weight', default=None, type=float,
                            help='the factor to use for weighted faces')

        return parser


class IcoMeshPredictionModule(IcoMeshDataModule):
    def __init__(self,
                 data_dir: str,
                 batch_size: int,
                 subdivisions: int,
                 data_internal_scale: float,
                 load_data_to_ram: bool = False,
                 vmask_thres: float = None,
                 vmask_weight: float = None,
                 fmask_thres: float = None,
                 fmask_weight: float = None):
        super().__init__(data_dir, batch_size, subdivisions, data_internal_scale, load_data_to_ram,
                         vmask_thres, vmask_weight, fmask_thres, fmask_weight)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        return self._create_val_dataloader()

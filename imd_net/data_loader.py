import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from pathlib import Path
from typing import Optional


class IcoMeshDataset(Dataset):
    """Data loader for IcoMesh datasets.

    Args:
        data_dir: path to data directory
        splits_dir: path to the directory that contains the dataset splits (training, validation, test)
        partition: train, val, test or all
        subdivisions: the number of subdivisions of the icosahedral mesh
        data_internal_scale: Internal scaling of the data
        load_data_to_ram: load the entire dataset to the RAM for training
        vmask_thres: the threshold dihedral angle above which vertices get weighted
        vmask_weight: the factor to use for weighted vertices
        fmask_thres: the threshold dihedral angle above which faces get weighted
        fmask_weight: the factor to use for weighted faces
    """

    def __init__(
        self,
        data_dir: str,
        splits_dir: str,
        partition: str,
        subdivisions: int,
        data_internal_scale: float,
        load_data_to_ram: bool = False,
        vmask_thres: Optional[float] = None,
        vmask_weight: Optional[float] = None,
        fmask_thres: Optional[float] = None,
        fmask_weight: Optional[float] = None,
    ):
        """
        Args:
            data_dir: path to data directory
            partition: train or test
        """
        dataset_path: Path = Path(data_dir)

        assert partition in ["train", "test", "val", "all"]
        assert subdivisions >= 0, "subdivision must be >= 0"
        assert dataset_path.exists() and dataset_path.is_dir(), f"{dataset_path} does not exist or is not a dir"

        # get file list
        splits_dir: Path = Path(splits_dir)
        partitions = ["train", "test", "val"] if partition == "all" else [partition]
        lines = []
        for part in partitions:
            split_file = splits_dir / f"{part}_split_{dataset_path.name}.txt"
            with open(split_file, "r") as f:
                lines.extend(f.readlines())

        self.flist = [dataset_path.joinpath(l.replace("\n", "").split(" ")[0]) for l in lines]

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

    def __getitem__(self, idx: int):
        return self.loaded_items[idx] or self._loaditem(idx)

    def _loaditem(self, idx: int):
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


class IcoMeshDataModule(L.LightningDataModule):
    """Lightning data module for IcoMesh datasets.

    Args:
        data_dir: path to data directory
        splits_dir: path to the directory that contains the dataset splits (training, validation, test)
        batch_size: the batch size
        subdivisions: the number of subdivisions of the icosahedral mesh
        data_internal_scale: Internal scaling of the data
        load_data_to_ram: load the entire dataset to the RAM for training
        vmask_thres: the threshold dihedral angle above which vertices get weighted
        vmask_weight: the factor to use for weighted vertices
        fmask_thres: the threshold dihedral angle above which faces get weighted
        fmask_weight: the factor to use for weighted faces
    """

    def __init__(
        self,
        data_dir: str,
        splits_dir: str,
        batch_size: int,
        subdivisions: int,
        data_internal_scale: float,
        load_data_to_ram: bool = False,
        vmask_thres: Optional[float] = None,
        vmask_weight: Optional[float] = None,
        fmask_thres: Optional[float] = None,
        fmask_weight: Optional[float] = None,
    ):
        super().__init__()

        # unpack input
        data_dir: Path = Path(data_dir)

        # check sanity of input
        assert subdivisions >= 0, "subdivision must be >= 0"
        assert data_dir.exists() and data_dir.is_dir(), f"{data_dir} does not exist or is not a dir"

        self.data_dir = data_dir
        self.splits_dir = splits_dir
        self.batch_size = batch_size
        self.subdivisions = subdivisions
        self.data_internal_scale = data_internal_scale
        self.load_data_to_ram = load_data_to_ram
        self.num_workers = 0 if self.load_data_to_ram else os.cpu_count()
        self.vmask_thres = vmask_thres
        self.vmask_weight = vmask_weight
        self.fmask_thres = fmask_thres
        self.fmask_weight = fmask_weight

    def prepare_data(self, *args, **kwargs):
        pass

    def _create_dataset(self, partition: str) -> IcoMeshDataset:
        return IcoMeshDataset(
            str(self.data_dir),
            self.splits_dir,
            partition,
            self.subdivisions,
            self.data_internal_scale,
            self.load_data_to_ram,
            self.vmask_thres,
            self.vmask_weight,
            self.fmask_thres,
            self.fmask_weight,
        )

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.data_train = self._create_dataset("train")
            self.data_val = self._create_dataset("val")

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.data_test = self._create_dataset("val")

        if stage == "predict":
            self.data_predict = self._create_dataset("all")

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.data_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )


import numpy as np
import torch
from pathlib import Path
from lightning.pytorch.callbacks import Callback
from typing import List
from .metrics import facenormals_angles, p2p_dist, StatsMeterLazy


class SavePredictionsCallback(Callback):
    def __init__(self, save_dir: str):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.final_save_dir = None
        self.flist = None

        assert self.save_dir.exists() and self.save_dir.is_dir(), "{} is not a dir".format(self.save_dir)

    def on_test_epoch_start(self, trainer, pl_module):
        self.flist = trainer.test_dataloaders.dataset.flist
        folder_name = self.flist[0].parent.name
        self.final_save_dir = self.save_dir / f"{folder_name}_IDN_npz"
        assert not self.final_save_dir.exists(), f"{self.final_save_dir} already exists"
        self.final_save_dir.mkdir()

    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # unpack batch
        _, targets, idx = batch

        # transform to vertices
        v = pl_module.reverse_internal_scale(pl_module.icomap2vertices(outputs)).cpu().numpy()
        v_target = pl_module.reverse_internal_scale(targets[0]).cpu().numpy()

        for i in range(v.shape[0]):
            # get original file name and construct output filename
            f_name = self.flist[idx[i]]
            f_out_absolute = self.final_save_dir / f"{f_name.stem}_denoised.npz"

            # load vertex and face masks
            v_mask = np.load(f_name)["v_max_dangles"]
            f_mask = np.load(f_name)["f_max_dangles"]
            data_mean = np.load(f_name)["data_mean"]
            data_scale = np.load(f_name)["data_scale"]

            np.savez(
                f_out_absolute,
                data=v[i].transpose(),
                targets=v_target[i].transpose(),
                v_max_dangles=v_mask,
                f_max_dangles=f_mask,
                data_mean=data_mean,
                data_scale=data_scale,
            )


class SaveMetricsCallback(Callback):
    def __init__(self, save_file: str):
        super().__init__()
        self.save_file = Path(save_file)
        assert not self.save_file.exists(), f"{save_file} already exists"

        self.flist = None
        self.header = "name,dist_mean,dist_max,dist_min,dist_std,angdiff_mean,angdiff_max,angdiff_min,angdiff_std"
        self.fnames = []
        self.dist_mean = np.array([])
        self.dist_max = np.array([])
        self.dist_min = np.array([])
        self.dist_std = np.array([])
        self.angdiff_mean = np.array([])
        self.angdiff_max = np.array([])
        self.angdiff_min = np.array([])
        self.angdiff_std = np.array([])

    def on_test_epoch_start(self, trainer, pl_module):
        self.flist = trainer.test_dataloaders.dataset.flist

    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # unpack batch
        _, targets, idx = batch

        # transform to vertices
        v = pl_module.reverse_internal_scale(pl_module.icomap2vertices(outputs))
        v_target = pl_module.reverse_internal_scale(targets[0].to(v.device))

        dist = p2p_dist(v, v_target).cpu().numpy()
        angdiff = facenormals_angles(v, v_target, pl_module.faces).cpu().numpy()

        # append file names
        self.fnames.extend([self.flist[idx[i]].name for i in range(idx.numel())])

        # compute and append distance metrics
        self.dist_mean = np.concatenate((self.dist_mean, dist.mean(-1)))
        self.dist_max = np.concatenate((self.dist_max, dist.max(-1)))
        self.dist_min = np.concatenate((self.dist_min, dist.min(-1)))
        self.dist_std = np.concatenate((self.dist_std, dist.std(-1)))

        # compute and append angular difference metrics
        self.angdiff_mean = np.concatenate((self.angdiff_mean, angdiff.mean(-1)))
        self.angdiff_max = np.concatenate((self.angdiff_max, angdiff.max(-1)))
        self.angdiff_min = np.concatenate((self.angdiff_min, angdiff.min(-1)))
        self.angdiff_std = np.concatenate((self.angdiff_std, angdiff.std(-1)))

    def on_test_epoch_end(self, trainer, pl_module):
        result = np.column_stack(
            (
                self.fnames,
                self.dist_mean,
                self.dist_max,
                self.dist_min,
                self.dist_std,
                self.angdiff_mean,
                self.angdiff_max,
                self.angdiff_min,
                self.angdiff_std,
            )
        )
        np.savetxt(self.save_file, result, delimiter=",", fmt="%s", header=self.header)


class LogMeshCallback(Callback):
    def __init__(self, log_mesh_epoch: int, log_mesh_samples: List[int]):
        self.log_mesh_epoch = log_mesh_epoch
        self.log_mesh_samples = log_mesh_samples

        self.base_stats = None
        self.base_p2p_metric = StatsMeterLazy()
        self.base_angular_metric = StatsMeterLazy()

        self.inputs = None
        self.targets = None

    def log_mesh(self, v, v_target, max_distance, global_step, pl_module):
        # create colors from distances
        color_p2pdist = p2p_dist(v, v_target)
        color_p2pdist = ((color_p2pdist.clamp(max=max_distance) / max_distance) * 255).unsqueeze(-1)
        color_p2pdist = torch.cat((color_p2pdist, torch.zeros_like(color_p2pdist).expand(-1, -1, 2)), dim=-1)

        pl_module.logger.experiment.add_mesh(
            "examples",
            vertices=v,
            colors=color_p2pdist,
            faces=pl_module.faces.unsqueeze(0).expand([v.shape[0], -1, -1]),
            global_step=global_step,
        )

    def setup(self, trainer, pl_module, stage: str) -> None:
        if self.base_p2p_metric:
            self.base_p2p_metric.to(device=pl_module.device)
        if self.base_angular_metric:
            self.base_angular_metric.to(device=pl_module.device)

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        if self.base_stats is None:
            # reverse internal scale for metric computations
            v_target = batch[1][0].to(device=pl_module.device)
            v_target = pl_module.reverse_internal_scale(v_target)

            # reconstruct vertices for input
            v_input = batch[0].to(device=pl_module.device)
            v_input = pl_module.icomap2vertices(v_input)
            v_input = pl_module.reverse_internal_scale(v_input)
            self.base_p2p_metric(p2p_dist(v_input, v_target).flatten())
            self.base_angular_metric(
                facenormals_angles(v_input, v_target, pl_module.faces, return_valid_mask=True).flatten()
            )

            # store the meshes in this callback for later processing
            if batch_idx == 0:
                self.inputs = batch[0][self.log_mesh_samples]
                self.targets = batch[1][0][self.log_mesh_samples]

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.base_stats is None:
            # log base metrics
            self.base_stats = {
                "dist": self.base_p2p_metric.compute(),
                "angle": self.base_angular_metric.compute(),
            }
            # free memory
            self.base_p2p_metric = None
            self.base_angular_metric = None

        # log meshes
        if pl_module.current_epoch % self.log_mesh_epoch == 0:
            # reverse internal scale for metric computations
            v_target = self.targets.to(device=pl_module.device)
            v_target = pl_module.reverse_internal_scale(v_target)

            v_pred = self.inputs.to(device=pl_module.device)
            v_pred = pl_module(v_pred)
            v_pred = pl_module.icomap2vertices(v_pred)
            v_pred = pl_module.reverse_internal_scale(v_pred)

            self.log_mesh(
                v_pred,
                v_target,
                self.base_stats["dist"]["mean"] + 2 * self.base_stats["dist"]["std"],
                pl_module.current_epoch,
                pl_module,
            )

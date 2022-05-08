from argparse import ArgumentParser
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import numpy as np

from module import IcoDenoiseNet, CustomModelCheckpoint
from loader import IcoMeshPredictionModule
import mesh.metrics as met


class SavePredictionsCallback(Callback):
    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.final_save_dir = None
        self.flist = None

        assert self.save_dir.exists() and self.save_dir.is_dir(), '{} is not a dir'.format(self.save_dir)

    def on_test_epoch_start(self, trainer, pl_module):
        self.flist = pl_module.test_dataloader().dataset.flist
        folder_name = self.flist[0].parent.name
        self.final_save_dir = self.save_dir / "{}_IDN_npz".format(folder_name)
        assert not self.final_save_dir.exists(), '{} already exists'.format(self.final_save_dir)
        self.final_save_dir.mkdir()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # unpack batch
        _, targets, idx = batch

        # # create predictions
        # data_hat = pl_module(data.to(pl_module.device, non_blocking=True))

        # transform to vertices
        v = pl_module.reverse_internal_scale(pl_module.icomap2vertices(outputs)).cpu().numpy()
        v_target = pl_module.reverse_internal_scale(targets[0]).numpy()

        for i in range(v.shape[0]):
            # get original file name and construct output filename
            f_name = self.flist[idx[i]]
            f_out_absolute = self.final_save_dir / "{}_denoised.npz".format(f_name.stem)

            # load vertex and face masks
            v_mask = np.load(f_name)["v_max_dangles"]
            f_mask = np.load(f_name)["f_max_dangles"]
            data_mean = np.load(f_name)["data_mean"]
            data_scale = np.load(f_name)["data_scale"]

            np.savez(f_out_absolute, data=v[i].transpose(), targets=v_target[i].transpose(),
                     v_max_dangles=v_mask, f_max_dangles=f_mask,
                     data_mean=data_mean, data_scale=data_scale)


class SaveMetricsCallback(Callback):
    def __init__(self, save_file):
        super().__init__()
        self.save_file = Path(save_file)
        assert not self.save_file.exists(), '{} already exists'.format(save_file)

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
        self.flist = pl_module.test_dataloader().dataset.flist

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # unpack batch
        _, targets, idx = batch

        # transform to vertices
        v = pl_module.reverse_internal_scale(pl_module.icomap2vertices(outputs))
        v_target = pl_module.reverse_internal_scale(targets[0].to(v.device))

        dist = met.p2p_dist(v, v_target).cpu().numpy()
        angdiff = met.facenormals_angles(v, v_target, pl_module.faces).cpu().numpy()

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
        result = np.column_stack((self.fnames,
                                  self.dist_mean, self.dist_max, self.dist_min, self.dist_std,
                                  self.angdiff_mean, self.angdiff_max, self.angdiff_min, self.angdiff_std))
        np.savetxt(self.save_file, result, delimiter=",", fmt="%s", header=self.header)


def main(args):
    # create datamodule
    dm = IcoMeshPredictionModule.from_argparse_args(args)

    # create model
    model = IcoDenoiseNet.load_from_checkpoint(args.resume_from_checkpoint)

    callbacks = []
    if args.save_predictions is not None:
        callbacks.append(SavePredictionsCallback(save_dir=args.save_predictions))
    if args.save_metrics is not None:
        callbacks.append(SaveMetricsCallback(save_file=args.save_metrics))

    # create trainer
    trainer = pl.Trainer.from_argparse_args(args, logger=False, checkpoint_callback=False, callbacks=callbacks)

    # test
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    parser = ArgumentParser()

    # define run specific arguments
    parser.add_argument("--save_predictions", default=None, type=str,
                        help='store all predictions as new dataset at given path')
    parser.add_argument("--save_metrics", default=None, type=str,
                        help='compute metrics on predictions and save as csv')

    # add params from Trainer
    parser = pl.Trainer.add_argparse_args(parser)
    # add params from model
    parser = IcoDenoiseNet.add_model_specific_args(parser)
    # add params from datamodule
    parser = IcoMeshPredictionModule.add_argparse_args(parser)

    args = parser.parse_args()
    main(args)

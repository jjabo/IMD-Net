from argparse import ArgumentParser
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import Optional, Union, List

from loss import MeshLoss
from icocnn.utils.ico_geometry import icomap_2_vertices, icomap_getPolesNeighborIndices, get_ico_faces, \
    get_edges_from_faces
from model import ConfigurableIcoResUNet
import utils.metrics as met


class IcoDenoiseNet(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        # create the network
        self.net = ConfigurableIcoResUNet(self.hparams.corner_mode,
                                          self.hparams.subdivisions,
                                          self.hparams.encoder_depth,
                                          self.hparams.layer_init_features,
                                          self.hparams.layer_feature_factor)
        # store trainable parameters
        kwargs['trainable_params'] = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        self.save_hyperparameters()

        # logging
        self.example_input_array = torch.rand([
            self.hparams.batch_size if "batch_size" in self.hparams else 4,
            3, 5 * (2 ** self.hparams.subdivisions), 2 ** (self.hparams.subdivisions + 1)])

        # loss function
        self.loss_function = MeshLoss(
            self.hparams.loss_pos_weight, self.hparams.loss_nor_weight, self.hparams.loss_cur_weight,
            # self.hparams.loss_use_lap_not_emc
        )

        # initialize properties and indices for icocnn nets
        # indices for averaging  north and southpole vertices from their neighbors
        corner_src_x, corner_src_y = icomap_getPolesNeighborIndices(self.hparams.subdivisions)
        self.register_buffer('corner_src_y', corner_src_y)
        self.register_buffer('corner_src_x', corner_src_x)
        # faces, faces adjacent to edges and vertices that compose edges
        ico_faces = get_ico_faces(self.hparams.subdivisions)
        edge_vertices, edge_faces = get_edges_from_faces(ico_faces)
        self.register_buffer('faces', torch.from_numpy(ico_faces))
        self.register_buffer('edge_vertices', torch.from_numpy(edge_vertices))
        self.register_buffer('edge_faces', torch.from_numpy(edge_faces))

    def forward(self, x):
        x_hat = self.net(x)
        return x_hat

    def step(self, batch):
        data, targets, idx = batch

        # feed data to net
        data_hat = self(data)

        # reconstruct vertices for each output
        v_hat = self.icomap2vertices(data_hat)

        # compute loss
        loss, loss_logs = self.loss_function(v_hat, targets[0], targets[1],
                                             self.faces, self.edge_vertices, self.edge_faces)

        return loss, loss_logs, v_hat, targets

    def icomap2vertices(self, icomap):
        if isinstance(icomap, tuple):
            v = tuple([icomap_2_vertices(d_i, self.corner_src_x, self.corner_src_y) for d_i in icomap])
        else:
            v = icomap_2_vertices(icomap, self.corner_src_x, self.corner_src_y)
        return v

    def reverse_internal_scale(self, v):
        return v / self.hparams.data_internal_scale

    def training_step(self, batch, batch_idx):
        loss, logs, _, _ = self.step(batch)

        if self.global_step % self.trainer.log_every_n_steps == 0:
            self.log_scalars('Loss', 'trn', logs)

        return loss

    def validation_step(self, batch, batch_idx):
        _, loss_logs, v_pred, targets = self.step(batch)

        # reverse internal scale for metric computations
        v_target = self.reverse_internal_scale(targets[0])
        v_pred = self.reverse_internal_scale(v_pred)

        # metrics
        p2p_dist = met.p2p_dist(v_pred, v_target)
        angdiff = met.facenormals_angles(v_pred, v_target, self.faces, return_valid_mask=True)

        # update logging
        loss_logs.update({'batch_size': v_pred.shape[0],
                          'dist_mean': p2p_dist.mean(), 'dist_max': p2p_dist.max(),
                          'angdiff_mean': angdiff.mean(), 'angdiff_max': angdiff.max(),
                          })

        return loss_logs

    def test_step(self, batch, batch_idx):
        data, targets, idx = batch

        # feed data to net
        data_hat = self(data)

        return data_hat

    def validation_epoch_end(self, validation_step_outputs):
        num_samples = validation_step_outputs[0]['batch_size']
        loss_tot = validation_step_outputs[0]['loss_tot'] * num_samples
        loss_pos = validation_step_outputs[0]['loss_pos'] * num_samples
        loss_nor = validation_step_outputs[0]['loss_nor'] * num_samples
        loss_cur = validation_step_outputs[0]['loss_cur'] * num_samples
        dist_mean = validation_step_outputs[0]['dist_mean'] * num_samples
        dist_max = validation_step_outputs[0]['dist_max']
        angdiff_mean = validation_step_outputs[0]['angdiff_mean'] * num_samples
        angdiff_max = validation_step_outputs[0]['angdiff_max']
        for step_logs in validation_step_outputs[1:]:
            loss_tot += step_logs['loss_tot'] * step_logs['batch_size']
            loss_pos += step_logs['loss_pos'] * step_logs['batch_size']
            loss_nor += step_logs['loss_nor'] * step_logs['batch_size']
            loss_cur += step_logs['loss_cur'] * step_logs['batch_size']
            dist_mean += step_logs['dist_mean'] * step_logs['batch_size']
            angdiff_mean += step_logs['angdiff_mean'] * step_logs['batch_size']
            dist_max = dist_max if dist_max > step_logs['dist_max'] else step_logs['dist_max']
            angdiff_max = angdiff_max if angdiff_max > step_logs['angdiff_max'] else step_logs['angdiff_max']
            num_samples += step_logs['batch_size']

        # log losses and metrics
        self.log_scalars('Loss', 'val', {
            'loss_tot': loss_tot / num_samples, 'loss_pos': loss_pos / num_samples,
            'loss_nor': loss_nor / num_samples, 'loss_cur': loss_cur / num_samples,
        })
        self.log_scalars('Distance', 'val', {
            'dist': dist_mean / num_samples, 'dist_max': dist_max
        })
        self.log_scalars('Angle', 'val', {
            'angdiff': angdiff_mean / num_samples, 'angdiff_max': angdiff_max
        })

        # log losses for hparam search etc.
        self.log('val_loss', loss_tot / num_samples)
        self.log('dist', dist_mean / num_samples)
        self.log('angdiff', angdiff_mean / num_samples)

    def log_scalars(self, chart_tile, scalar_prefix, scalar_dict):
        self.logger.experiment.add_scalars(chart_tile,
                                           {f"{scalar_prefix}_{k}": v for k, v in scalar_dict.items()},
                                           global_step=self.global_step)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--corner_mode", default='average', type=str, choices=['average', 'zeros'],
                            help='Mode for computing corners of icosahedron')
        parser.add_argument('--subdivisions', default=5, type=int,
                            help='subdivisions of icosahedral input data')
        parser.add_argument('--encoder_depth', default=3, type=int,
                            help='number of layers in encoder')
        parser.add_argument('--layer_init_features', default=28, type=int,
                            help='number of features in first (and last) layer of encoder')
        parser.add_argument('--layer_feature_factor', default=1.6, type=float,
                            help='the factor by which number of features are increased in each encoder layer')
        # loss function params
        parser.add_argument('--loss_pos_weight', default=1.0, type=float,
                            help='the weight of the loss function on vertex positions')
        parser.add_argument('--loss_nor_weight', default=0, type=float,
                            help='the weight of the loss function on angular difference between face normals')
        parser.add_argument('--loss_cur_weight', default=10, type=float,
                            help='the weight of the loss function on mean curvature')
        # learning rate
        parser.add_argument("--learning_rate", default=4.5e-4, type=float,
                            help='the learning rate for optimizer')

        # params for internal handling of data
        parser.add_argument("--data_internal_scale", default=10000.0, type=float,
                            help='the scale that is applied internally to the input data')

        # params for logging
        parser.add_argument("--log_mesh_epoch", default=4, type=int,
                            help='on every x-th epoch meshes are logged')
        parser.add_argument("--log_mesh_samples", default=[0], type=int, nargs='+',
                            help='which meshes of the first validation batch should be logged')
        parser.add_argument("--log_dir", default='./logs', type=str,
                            help='the dir for logging')
        return parser


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(
            self,
            monitor: Optional[str] = None,
            hp_metrics: Optional[List[str]] = None,
            verbose: bool = False,
            save_top_k: int = 1,
            save_weights_only: bool = False,
            mode: str = "auto",
            period: int = 1,
            prefix: str = "",
            dirpath: Optional[Union[str, Path]] = None,
            filename: Optional[str] = None):
        super().__init__(
            monitor=monitor, verbose=verbose, save_last=True,
            save_top_k=save_top_k, save_weights_only=save_weights_only, mode=mode,
            period=period, prefix=prefix, dirpath=dirpath, filename=filename)

        self.hp_metrics = hp_metrics

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)

        metrics = self._monitor_candidates(trainer)
        if self.best_model_score and isinstance(self.best_model_score, torch.Tensor):
            self.best_model_score = self.best_model_score.cuda(metrics.get(self.monitor).get_device()) \
                if metrics.get(self.monitor).is_cuda else self.best_model_score

        if self.best_model_score == metrics.get(self.monitor):
            # remove keys that are not relevant
            for key in list(metrics.keys()):
                if key not in self.hp_metrics:
                    del metrics[key]

            # rename metrics to avoid overwriting original metric
            metrics = {f"m_{k}": v for k, v in metrics.items()}

            pl_module.logger.log_hyperparams(pl_module.hparams, metrics)

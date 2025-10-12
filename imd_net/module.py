from pathlib import Path
import torch
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from typing import Optional, Union, List, Literal

from loss import MeshLoss
from icocnn.utils.ico_geometry import icomap_2_vertices, icomap_getPolesNeighborIndices, get_ico_faces, \
    get_edges_from_faces
from components import ConfigurableIcoResUNet
import utils.metrics as met

class IcoDenoiseNet(pl.LightningModule):
    """A denoising network for icosahedral meshes based on a U-Net architecture with group equivariant convolutions.

    Args:
        corner_mode: Mode for computing corners of icosahedron. Either 'average' or 'zeros'.
        batch_size: The batch size for training, needed here for tensorboard logging
        subdivisions: Subdivisions of icosahedral input data.
        encoder_depth: Number of layers in encoder.
        layer_init_features: Number of features in first (and last) layer of encoder.
        layer_feature_factor: The factor by which number of features are increased in each encoder layer.
        loss_pos_weight: The weight of the loss function on vertex positions.
        loss_nor_weight: The weight of the loss function on angular difference between face normals.
        loss_cur_weight: The weight of the loss function on mean curvature.
        learning_rate: The learning rate for optimizer.
        data_internal_scale: The scale that is applied internally to the input data.
        log_mesh_epoch: On every x-th epoch meshes are logged.
        log_mesh_samples: Which meshes of the first validation batch should be logged.
        log_dir: The dir for logging.
    """
    def __init__(self,
                 corner_mode: Literal['average', 'zeros'] = 'average',
                 batch_size: int = 8,
                 subdivisions: int = 6,
                 encoder_depth: int = 3,
                 layer_init_features: int = 28,
                 layer_feature_factor: float = 1.6,
                 loss_pos_weight: float = 1.0,
                 loss_nor_weight: float = 0,
                 loss_cur_weight: float = 10.0,
                 learning_rate: float = 4.5e-4,
                 data_internal_scale: float = 10000.0,
                 log_mesh_epoch: int = 4,
                 log_mesh_samples: List[int] = [0],
                 log_dir: str = './logs'):
        super().__init__()
        self.save_hyperparameters()
        self.corner_mode = corner_mode
        self.batch_size = batch_size
        self.subdivisions = subdivisions
        self.encoder_depth = encoder_depth
        self.layer_init_features = layer_init_features
        self.layer_feature_factor = layer_feature_factor
        self.loss_pos_weight = loss_pos_weight
        self.loss_nor_weight = loss_nor_weight
        self.loss_cur_weight = loss_cur_weight
        self.learning_rate = learning_rate
        self.data_internal_scale = data_internal_scale
        self.log_mesh_epoch = log_mesh_epoch
        self.log_mesh_samples = log_mesh_samples
        self.log_dir = log_dir

        # create the network
        self.net = ConfigurableIcoResUNet(self.corner_mode,
                                          self.subdivisions,
                                          self.encoder_depth,
                                          self.layer_init_features,
                                          self.layer_feature_factor)
        # store trainable parameters
        self.trainable_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        self.save_hyperparameters()

        self.validation_step_outputs = []

        # tensorboard needs this for logging the model graph
        self.example_input_array = torch.rand([
            self.batch_size, 3, 5 * (2**self.subdivisions),
            2**(self.subdivisions + 1)
        ])

        # loss function
        self.loss_function = MeshLoss(self.loss_pos_weight, self.loss_nor_weight, self.loss_cur_weight)

        # initialize properties and indices for icocnn nets
        # indices for averaging north and southpole vertices from their neighbors
        corner_src_x, corner_src_y = icomap_getPolesNeighborIndices(self.subdivisions)
        self.register_buffer('corner_src_y', corner_src_y)
        self.register_buffer('corner_src_x', corner_src_x)
        # faces, faces adjacent to edges and vertices that compose edges
        ico_faces = get_ico_faces(self.subdivisions)
        edge_vertices, edge_faces = get_edges_from_faces(ico_faces)
        self.register_buffer('faces', torch.from_numpy(ico_faces))
        self.register_buffer('edge_vertices', torch.from_numpy(edge_vertices))
        self.register_buffer('edge_faces', torch.from_numpy(edge_faces))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_hat = self.net(x)
        return x_hat

    def step(self, batch: tuple) -> tuple:
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
        return v / self.data_internal_scale

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

        # store values for epoch end logging
        self.validation_step_outputs.append(loss_logs)

        return loss_logs

    def test_step(self, batch, batch_idx):
        data, targets, idx = batch

        # feed data to net
        data_hat = self(data)

        return data_hat

    def on_validation_epoch_end(self):
        num_samples = self.validation_step_outputs[0]['batch_size']
        loss_tot = self.validation_step_outputs[0]['loss_tot'] * num_samples
        loss_pos = self.validation_step_outputs[0]['loss_pos'] * num_samples
        loss_nor = self.validation_step_outputs[0]['loss_nor'] * num_samples
        loss_cur = self.validation_step_outputs[0]['loss_cur'] * num_samples
        dist_mean = self.validation_step_outputs[0]['dist_mean'] * num_samples
        dist_max = self.validation_step_outputs[0]['dist_max']
        angdiff_mean = self.validation_step_outputs[0]['angdiff_mean'] * num_samples
        angdiff_max = self.validation_step_outputs[0]['angdiff_max']
        for step_logs in self.validation_step_outputs[1:]:
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

        self.validation_step_outputs.clear()

    def log_scalars(self, chart_tile, scalar_prefix, scalar_dict):
        self.logger.experiment.add_scalars(chart_tile,
                                           {f"{scalar_prefix}_{k}": v for k, v in scalar_dict.items()},
                                           global_step=self.global_step)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)
        return optimizer

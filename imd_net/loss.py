import torch
import torch.nn as nn

from mesh.utils import compute_face_normals, mean_curvature_edge
# from icocnn.utils.ico_geometry import get_ico_faces
# from mesh.utils import sample_faces_batch, sample_point_from_face_batch
# import mesh.metrics as met
# from icocnn.utils.ico_geometry import get_edges_from_faces


# class PointAndFaceAndLB_Loss(nn.Module):
#     def __init__(self, subdivisions, factor_pos, factor_nor, factor_lap, use_emc_loss, use_emc_loss_along_enormal):
#         super(PointAndFaceAndLB_Loss, self).__init__()
#
#         self.factor_pos = factor_pos
#         self.factor_nor = factor_nor
#         self.factor_lap = factor_lap
#         self.use_emc_loss = use_emc_loss
#         self.use_emc_loss_along_enormal = use_emc_loss_along_enormal
#
#         self.loss_posMSE = nn.MSELoss(reduction='none')
#         self.loss_norCos = nn.CosineSimilarity(dim=1)
#         self.loss_lap = nn.MSELoss()
#
#         # index buffers for icosahedral corner averaging
#         base_height = 2 ** subdivisions
#         top_corner_src_y = torch.arange(5) * base_height
#         top_corner_src_x = torch.tensor([0])
#         bottom_corner_src_y = torch.arange(1, 6) * base_height - 1
#         bottom_corner_src_x = torch.tensor([-1])
#         corner_src_y = torch.stack((top_corner_src_y, bottom_corner_src_y))
#         corner_src_x = torch.stack((top_corner_src_x, bottom_corner_src_x))
#         self.register_buffer('corner_src_y', corner_src_y)
#         self.register_buffer('corner_src_x', corner_src_x)
#
#         # faces
#         ico_faces = get_ico_faces(subdivisions)
#         edge_vertices, edge_faces = get_edges_from_faces(ico_faces)
#         self.register_buffer('ico_faces', torch.from_numpy(ico_faces))
#         self.register_buffer('edge_vertices', torch.from_numpy(edge_vertices))
#         self.register_buffer('edge_faces', torch.from_numpy(edge_faces))
#
#         self.last_loss_mse = 0
#         self.last_loss_nor = 0
#         self.last_loss_lap = 0
#         self.last_loss_total = 0
#
#     def forward(self, inputs, target):
#         # unpack targets
#         target_vals = target[0]
#         feature_masks = target[1]
#
#         # VERTEX POSITIONAL LOSS
#         # compute pole vertices
#         v_tmp = inputs.view(inputs.shape[0], inputs.shape[1], -1)
#         v_poles = torch.mean(inputs[:, :, self.corner_src_y, self.corner_src_x], -1)
#         data_v = torch.cat((v_tmp, v_poles), dim=2).transpose(1, 2).contiguous()
#
#         # prepare targets
#         targets_v = target_vals
#
#         # compute loss
#         l_posMSE = torch.sum(feature_masks * self.loss_posMSE(data_v, targets_v)) / torch.sum(feature_masks)
#         self.last_loss_mse = l_posMSE.item()
#
#         # FACE NORMAL LOSS
#         # compute face normals
#         data_f_normals = compute_face_normals(data_v, self.ico_faces)
#         # compute groundtruth face normals
#         targets_f_normals, valid_tnormals_mask = compute_face_normals(targets_v, self.ico_faces, return_valid_mask=True)
#         # compute loss of face normals
#         l_norCos = torch.mean(1 - self.loss_norCos(data_f_normals[valid_tnormals_mask], targets_f_normals[valid_tnormals_mask]))
#         self.last_loss_nor = l_norCos.item()
#
#         # CURVATURE LOSS
#         if self.use_emc_loss:
#             # compute mean edge curvatures
#             mce, valid_mask = mean_curvature_edge(data_v, self.ico_faces, self.edge_vertices, self.edge_faces, project_along_edge_normal=self.use_emc_loss_along_enormal)
#             # compute mean curvature for edges
#             targets_mce, targets_valid_mask = mean_curvature_edge(targets_v, self.ico_faces, self.edge_vertices,
#                                                                   self.edge_faces, project_along_edge_normal=self.use_emc_loss_along_enormal)
#
#             combined_valid_mask = valid_mask & targets_valid_mask
#             l_lap = 0
#             if mce[combined_valid_mask].numel() > 0:
#                 l_lap = self.loss_lap(mce[combined_valid_mask], targets_mce[combined_valid_mask])
#         else:
#             # compute vertex laplacians
#             data_lap, valid_mask = compute_laplaceBeltrami(data_v, self.ico_faces, return_valid_mask=True)
#             targets_lap, target_valid_mask = compute_laplaceBeltrami(targets_v, self.ico_faces,
#                                                                          return_valid_mask=True)
#             combined_valid_mask = valid_mask & target_valid_mask
#             l_lap = self.loss_lap(data_lap[combined_valid_mask], targets_lap[combined_valid_mask])
#
#         self.last_loss_lap = l_lap.item()
#
#         loss = self.factor_pos * l_posMSE + self.factor_nor * l_norCos + self.factor_lap * l_lap
#         self.last_loss_total = loss.item()
#
#         return loss
#
#     def get_last_losses(self):
#         return self.last_loss_mse, self.last_loss_nor, self.last_loss_lap, self.last_loss_total


# class Cascaded_Loss(nn.Module):
#     def __init__(self, subdivisions, cascaded_depth, cascaded_weights, factor_pos, factor_nor, factor_lap, use_emc_loss, use_emc_loss_along_enormal):
#         super(Cascaded_Loss, self).__init__()
#
#         self.cascaded_depth = cascaded_depth
#         self.cascaded_weights = cascaded_weights
#         self.active_module_indices = list(range(cascaded_depth))
#
#         self.loss_list = nn.ModuleList()
#         for i in range(self.cascaded_depth):
#             self.loss_list.append(PointAndFaceAndLB_Loss(subdivisions, factor_pos[i], factor_nor[i], factor_lap[i], use_emc_loss, use_emc_loss_along_enormal))
#
#         # index buffers for icosahedral corner averaging
#         base_height = 2 ** subdivisions
#         top_corner_src_y = torch.arange(5) * base_height
#         top_corner_src_x = torch.tensor([0])
#         bottom_corner_src_y = torch.arange(1, 6) * base_height - 1
#         bottom_corner_src_x = torch.tensor([-1])
#         corner_src_y = torch.stack((top_corner_src_y, bottom_corner_src_y))
#         corner_src_x = torch.stack((top_corner_src_x, bottom_corner_src_x))
#         self.register_buffer('corner_src_y', corner_src_y)
#         self.register_buffer('corner_src_x', corner_src_x)
#
#         # faces
#         ico_faces = get_ico_faces(subdivisions)
#         self.register_buffer('ico_faces', torch.from_numpy(ico_faces))
#
#         self.subnets_last_loss_mse = []
#         self.subnets_last_loss_nor = []
#         self.subnets_last_loss_lap = []
#         self.subnets_last_loss_tot = []
#         self.last_loss_mse = 0
#         self.last_loss_nor = 0
#         self.last_loss_lap = 0
#         self.last_loss_tot = 0
#
#     def forward(self, inputs, target):
#         self.subnets_last_loss_mse = [None] * self.cascaded_depth
#         self.subnets_last_loss_nor = [None] * self.cascaded_depth
#         self.subnets_last_loss_lap = [None] * self.cascaded_depth
#         self.subnets_last_loss_tot = [None] * self.cascaded_depth
#         self.last_loss_mse = 0
#         self.last_loss_nor = 0
#         self.last_loss_lap = 0
#         self.last_loss_tot = 0
#         loss_total = 0
#
#         for active_idx in range(self.cascaded_depth):
#             if active_idx in self.active_module_indices:
#                 loss_total = loss_total + \
#                              self.loss_list[active_idx](inputs[active_idx], target) * self.cascaded_weights[active_idx]
#                 self.subnets_last_loss_mse[active_idx] = self.loss_list[active_idx].last_loss_mse
#                 self.subnets_last_loss_nor[active_idx] = self.loss_list[active_idx].last_loss_nor
#                 self.subnets_last_loss_lap[active_idx] = self.loss_list[active_idx].last_loss_lap
#                 self.subnets_last_loss_tot[active_idx] = self.loss_list[active_idx].last_loss_total
#
#                 # record losses
#                 self.last_loss_mse += self.subnets_last_loss_mse[active_idx] * self.cascaded_weights[active_idx]
#                 self.last_loss_nor += self.subnets_last_loss_nor[active_idx] * self.cascaded_weights[active_idx]
#                 self.last_loss_lap += self.subnets_last_loss_lap[active_idx] * self.cascaded_weights[active_idx]
#                 self.last_loss_tot += self.subnets_last_loss_tot[active_idx] * self.cascaded_weights[active_idx]
#
#         return loss_total
#
#     def get_last_losses(self):
#         return self.subnets_last_loss_mse, self.subnets_last_loss_nor,  self.subnets_last_loss_lap, \
#                self.subnets_last_loss_tot, self.last_loss_mse, self.last_loss_nor, self.last_loss_lap, \
#                self.last_loss_tot
#
#     def set_active_modules(self, active_modules_indices):
#         self.active_module_indices = active_modules_indices


class MeshLoss(nn.Module):
    def __init__(self, factor_pos, factor_nor, factor_cur,
                 # use_lap_not_emc
                 ):
        super(MeshLoss, self).__init__()

        self.factor_pos = factor_pos
        self.factor_nor = factor_nor
        self.factor_cur = factor_cur
        # self.use_lap_not_emc = use_lap_not_emc

        self.loss_fun_pos = nn.MSELoss(reduction='none')
        self.loss_fun_nor = nn.CosineSimilarity(dim=1)
        self.loss_fun_cur = nn.MSELoss()

    def forward(self, v, v_target, v_mask, faces, edge_vertices, edge_faces):
        # POSITIONAL MSE LOSS
        loss_pos = torch.sum(v_mask * self.loss_fun_pos(v, v_target)) / torch.sum(v_mask)

        # FACE NORMAL LOSS
        data_f_normals = compute_face_normals(v, faces)
        targets_f_normals, normals_mask = compute_face_normals(v_target, faces, return_valid_mask=True)
        # compute loss of face normals - ignore normals that have degenerate groundtruth faces
        loss_nor = torch.mean(1 - self.loss_fun_nor(data_f_normals[normals_mask], targets_f_normals[normals_mask]))

        # CURVATURE LOSS
        loss_cur = 0
        # if not self.use_lap_not_emc:
        #     # compute mean edge curvatures
        #     cur_vals, valid_mask = mean_curvature_edge(v, faces, edge_vertices, edge_faces)
        #     targets_cur_vals, targets_valid_mask = mean_curvature_edge(v_target, faces, edge_vertices, edge_faces)
        # else:
        #     # compute vertex laplacians
        #     cur_vals, valid_mask = compute_laplaceBeltrami(v, faces, return_valid_mask=True)
        #     targets_cur_vals, targets_valid_mask = compute_laplaceBeltrami(v_target, faces, return_valid_mask=True)

        cur_vals, valid_mask = mean_curvature_edge(v, faces, edge_vertices, edge_faces)
        targets_cur_vals, targets_valid_mask = mean_curvature_edge(v_target, faces, edge_vertices, edge_faces)
        combined_valid_mask = valid_mask & targets_valid_mask
        if combined_valid_mask.any():
            loss_cur = self.loss_fun_cur(cur_vals[combined_valid_mask], targets_cur_vals[combined_valid_mask])

        loss = self.factor_pos * loss_pos + self.factor_nor * loss_nor + self.factor_cur * loss_cur

        loss_log = {
            'loss_tot': loss.detach(),
            'loss_pos': loss_pos.detach(),
            'loss_nor': loss_nor.detach(),
            'loss_cur': loss_cur.detach(),
        }

        return loss, loss_log


# class CascadedMeshLoss(nn.Module):
#     def __init__(self, cnt_cascaded_nets, cascaded_weights, factor_pos, factor_nor, factor_lap, use_emc_loss, use_emc_loss_along_enormal):
#         super(CascadedMeshLoss, self).__init__()
#
#         self.cascaded_depth = cnt_cascaded_nets
#         self.cascaded_weights = cascaded_weights
#         self.active_module_indices = list(range(self.cascaded_depth))
#
#         self.loss_list = nn.ModuleList()
#         for i in range(self.cascaded_depth):
#             self.loss_list.append(
#                 MeshLoss(factor_pos[i], factor_nor[i], factor_lap[i], use_emc_loss, use_emc_loss_along_enormal))
#
#         self.subnets_last_loss_mse = []
#         self.subnets_last_loss_nor = []
#         self.subnets_last_loss_lap = []
#         self.subnets_last_loss_tot = []
#         self.last_loss_mse = 0
#         self.last_loss_nor = 0
#         self.last_loss_lap = 0
#         self.last_loss_tot = 0
#
#     def forward(self, v, v_target, vmask, faces, edge_vertices, edge_faces):
#         self.subnets_last_loss_mse = [None] * self.cascaded_depth
#         self.subnets_last_loss_nor = [None] * self.cascaded_depth
#         self.subnets_last_loss_lap = [None] * self.cascaded_depth
#         self.subnets_last_loss_tot = [None] * self.cascaded_depth
#         self.last_loss_mse = 0
#         self.last_loss_nor = 0
#         self.last_loss_lap = 0
#         self.last_loss_tot = 0
#         loss_total = 0
#
#         for active_idx in self.active_module_indices:
#             loss_total = loss_total + self.loss_list[active_idx](
#                 v[active_idx], v_target, vmask, faces, edge_vertices, edge_faces
#             ) * self.cascaded_weights[active_idx]
#             self.subnets_last_loss_mse[active_idx] = self.loss_list[active_idx].last_loss_mse
#             self.subnets_last_loss_nor[active_idx] = self.loss_list[active_idx].last_loss_nor
#             self.subnets_last_loss_lap[active_idx] = self.loss_list[active_idx].last_loss_lap
#             self.subnets_last_loss_tot[active_idx] = self.loss_list[active_idx].last_loss_total
#
#             # record losses
#             self.last_loss_mse += self.subnets_last_loss_mse[active_idx] * self.cascaded_weights[active_idx]
#             self.last_loss_nor += self.subnets_last_loss_nor[active_idx] * self.cascaded_weights[active_idx]
#             self.last_loss_lap += self.subnets_last_loss_lap[active_idx] * self.cascaded_weights[active_idx]
#             self.last_loss_tot += self.subnets_last_loss_tot[active_idx] * self.cascaded_weights[active_idx]
#
#         return loss_total
#
#     def get_last_losses(self):
#         return self.subnets_last_loss_mse, self.subnets_last_loss_nor,  self.subnets_last_loss_lap, \
#                self.subnets_last_loss_tot, self.last_loss_mse, self.last_loss_nor, self.last_loss_lap, \
#                self.last_loss_tot
#
#     def set_active_modules(self, active_modules_indices):
#         self.active_module_indices = active_modules_indices

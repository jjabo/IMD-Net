import torch
import torch.nn as nn
from utils.mesh import compute_face_normals, mean_curvature_edge


class MeshLoss(nn.Module):
    def __init__(self, factor_pos, factor_nor, factor_cur):
        super(MeshLoss, self).__init__()

        self.factor_pos = factor_pos
        self.factor_nor = factor_nor
        self.factor_cur = factor_cur

        self.loss_fun_pos = nn.MSELoss(reduction='none')
        self.loss_fun_nor = nn.CosineSimilarity(dim=1)
        self.loss_fun_cur = nn.MSELoss()

    def forward(self, v, v_target, v_mask, faces, edge_vertices, edge_faces):
        # POSITIONAL MSE LOSS
        loss_pos = torch.sum(v_mask * self.loss_fun_pos(v, v_target)) / torch.sum(v_mask)

        # FACE NORMAL LOSS
        data_f_normals = compute_face_normals(v, faces)
        targets_f_normals, normals_mask = compute_face_normals(v_target, faces, return_valid_mask=True)
        # compute loss of face normals - ignore normals that have degenerate ground truth faces
        loss_nor = torch.mean(1 - self.loss_fun_nor(data_f_normals[normals_mask], targets_f_normals[normals_mask]))

        # CURVATURE LOSS
        # compute mean edge curvatures
        loss_cur = 0
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

import torch


def compute_face_normals(v, f, eps=1e-10, return_valid_mask=False):
    # get vertices for face positions 0, 1, 2
    a = torch.index_select(v, dim=1, index=f[:, 0])
    b = torch.index_select(v, dim=1, index=f[:, 1])
    c = torch.index_select(v, dim=1, index=f[:, 2])

    # compute unnormalized (area weighted) face normals
    f_normals = torch.cross(b - a, c - a, dim=2)

    # normalize
    magnitude = torch.norm(f_normals, 2, dim=2, keepdim=True)
    if return_valid_mask:
        valid_mask = (magnitude >= eps).squeeze(dim=-1)
    magnitude = torch.clamp(magnitude, eps)
    f_normals = torch.div(f_normals, magnitude)

    if return_valid_mask:
        return f_normals, valid_mask
    return f_normals


def mean_curvature_edge(vertices: torch.Tensor, faces: torch.Tensor, edge_vertices: torch.Tensor,
                        edge_faces: torch.Tensor, project_along_edge_normal=True, eps=1e-10):

    # get face normals
    face_normals, fnormals_valid_mask = compute_face_normals(vertices, faces, return_valid_mask=True)

    # computed edge normal
    normal0 = face_normals[:, edge_faces[:, 0], :]
    normal1 = face_normals[:, edge_faces[:, 1], :]

    # compute mask to exclude edges that are part of invalid faces
    valid_mask = fnormals_valid_mask[:, edge_faces[:, 0]]
    valid_mask = valid_mask & fnormals_valid_mask[:, edge_faces[:, 1]]

    # compute dihedral angle
    angle_sin = torch.norm(torch.cross(normal0, normal1, dim=2), dim=2)
    angle_cos = torch.sum(normal0 * normal1, dim=2)
    dihedral_angles = torch.atan2(angle_sin, angle_cos).unsqueeze(-1)

    # compute edge length
    edge_length = torch.norm(vertices[:, edge_vertices[:, 0], :] - vertices[:, edge_vertices[:, 1], :], dim=2,
                             keepdim=True)

    # compute edge mean curvature
    mean_curv = 0.5 * edge_length * torch.cos(dihedral_angles * 0.5)

    # project edge mean curvature along edge normal
    if project_along_edge_normal:
        # compute edge normal
        edge_normals = normal0 + normal1

        # normalize edge normal and update valid_mask to exclude edges whose faces have angle pi
        magnitude = torch.norm(edge_normals, 2, dim=2, keepdim=True)
        valid_mask = valid_mask & (magnitude > eps).squeeze()  # edge normals can become zero if their angle is Pi
        magnitude = torch.clamp(magnitude, eps)
        edge_normals = torch.div(edge_normals, magnitude)

        # project edge mean curvature along edge normal
        mean_curv = mean_curv * edge_normals

    return mean_curv, valid_mask




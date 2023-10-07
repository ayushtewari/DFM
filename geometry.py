"""Multi-view geometry & proejction code.."""
import torch
from einops import rearrange, repeat


def symmetric_orthogonalization(x):
    # https://github.com/amakadia/svd_for_pose
    m = x.view(-1, 3, 3).type(torch.float)
    u, s, v = torch.svd(m)
    vt = torch.transpose(v, 1, 2)
    det = torch.det(torch.matmul(u, vt))
    det = det.view(-1, 1, 1)
    vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
    r = torch.matmul(u, vt)
    return r


def homogenize_points(points: torch.Tensor):
    """Appends a "1" to the coordinates of a (batch of) points of dimension DIM.

    Args:
        points: points of shape (..., DIM)

    Returns:
        points_hom: points with appended "1" dimension.
    """
    ones = torch.ones_like(points[..., :1], device=points.device)
    return torch.cat((points, ones), dim=-1)


def homogenize_vecs(vectors: torch.Tensor):
    """Appends a "0" to the coordinates of a (batch of) vectors of dimension DIM.

    Args:
        vectors: vectors of shape (..., DIM)

    Returns:
        vectors_hom: points with appended "0" dimension.
    """
    zeros = torch.zeros_like(vectors[..., :1], device=vectors.device)
    return torch.cat((vectors, zeros), dim=-1)


def unproject(
    xy_pix: torch.Tensor, z: torch.Tensor, intrinsics: torch.Tensor
) -> torch.Tensor:
    """Unproject (lift) 2D pixel coordinates x_pix and per-pixel z coordinate
    to 3D points in camera coordinates.

    Args:
        xy_pix: 2D pixel coordinates of shape (..., 2)
        z: per-pixel depth, defined as z coordinate of shape (..., 1)
        intrinscis: camera intrinscics of shape (..., 3, 3)

    Returns:
        xyz_cam: points in 3D camera coordinates.
    """
    xy_pix_hom = homogenize_points(xy_pix)
    xyz_cam = torch.einsum("...ij,...kj->...ki", intrinsics.inverse(), xy_pix_hom)
    xyz_cam *= z
    return xyz_cam


def transform_world2cam(
    xyz_world_hom: torch.Tensor, cam2world: torch.Tensor
) -> torch.Tensor:
    """Transforms points from 3D world coordinates to 3D camera coordinates.

    Args:
        xyz_world_hom: homogenized 3D points of shape (..., 4)
        cam2world: camera pose of shape (..., 4, 4)

    Returns:
        xyz_cam: points in camera coordinates.
    """
    world2cam = torch.inverse(cam2world)
    return transform_rigid(xyz_world_hom, world2cam)


def transform_cam2world(
    xyz_cam_hom: torch.Tensor, cam2world: torch.Tensor
) -> torch.Tensor:
    """Transforms points from 3D world coordinates to 3D camera coordinates.

    Args:
        xyz_cam_hom: homogenized 3D points of shape (..., 4)
        cam2world: camera pose of shape (..., 4, 4)

    Returns:
        xyz_world: points in camera coordinates.
    """
    return transform_rigid(xyz_cam_hom, cam2world)


def transform_rigid(xyz_hom: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Apply a rigid-body transform to a (batch of) points / vectors.

    Args:
        xyz_hom: homogenized 3D points of shape (..., 4)
        T: rigid-body transform matrix of shape (..., 4, 4)

    Returns:
        xyz_trans: transformed points.
    """
    return torch.einsum("...ij,...kj->...ki", T, xyz_hom)


def get_unnormalized_cam_ray_directions(
    xy_pix: torch.Tensor, intrinsics: torch.Tensor
) -> torch.Tensor:
    return unproject(
        xy_pix,
        torch.ones_like(xy_pix[..., :1], device=xy_pix.device),
        intrinsics=intrinsics,
    )


def get_world_rays(
    xy_pix: torch.Tensor, intrinsics: torch.Tensor, cam2world: torch.Tensor,
) -> torch.Tensor:
    # Get camera origin of camera 1
    cam_origin_world = cam2world[..., :3, -1]

    # Get ray directions in cam coordinates
    ray_dirs_cam = get_unnormalized_cam_ray_directions(xy_pix, intrinsics)
    ray_dirs_cam /= ray_dirs_cam.norm(dim=-1, keepdim=True)

    # Homogenize ray directions
    rd_cam_hom = homogenize_vecs(ray_dirs_cam)

    # Transform ray directions to world coordinates
    rd_world_hom = transform_cam2world(rd_cam_hom, cam2world)

    # Tile the ray origins to have the same shape as the ray directions.
    # Currently, ray origins have shape (batch, 3), while ray directions have shape
    cam_origin_world = repeat(
        cam_origin_world, "b ch -> b num_rays ch", num_rays=ray_dirs_cam.shape[1]
    )

    # Return tuple of cam_origins, ray_world_directions
    return cam_origin_world, rd_world_hom[..., :3]


def get_world_rays_top_down(
    xy_pix: torch.Tensor, intrinsics: torch.Tensor, cam2world: torch.Tensor,
) -> torch.Tensor:
    # Get camera origin of camera 1
    cam_origin_world = cam2world[..., :3, -1]
    # print(cam_origin_world.shape, "cam_origin_world.shape")

    # move camera origin to top down view
    # cam_origin_world[..., :3] = 0.0
    # cam_origin_world[..., 1] = -1.0

    # Get ray directions in cam coordinates
    ray_dirs_cam = get_unnormalized_cam_ray_directions(xy_pix, intrinsics)
    ray_dirs_cam /= ray_dirs_cam.norm(dim=-1, keepdim=True)
    # print(ray_dirs_cam.shape, "ray_dirs_cam")
    # print(
    #     "ray_dirs_cam.shape", ray_dirs_cam.shape,
    # )

    ray_dirs_cam[..., :3] = 0.0
    ray_dirs_cam[..., 1] = 1.0

    # Homogenize ray directions
    rd_cam_hom = homogenize_vecs(ray_dirs_cam)

    # Transform ray directions to world coordinates
    rd_world_hom = transform_cam2world(rd_cam_hom, cam2world)

    # Tile the ray origins to have the same shape as the ray directions.
    # Currently, ray origins have shape (batch, 3), while ray directions have shape
    cam_origin_world = repeat(
        cam_origin_world, "b ch -> b num_rays ch", num_rays=ray_dirs_cam.shape[1]
    )

    # xy_pix = get_opencv_pixel_coordinates(64, 64)
    # xy_pix = rearrange(xy_pix, "h w c -> () (h w) c")
    xz = (xy_pix * 2.0 - 1.0) * 3.5
    # chunk xz into x and z
    x = xz[..., 0]
    z = xz[..., 1]
    y = torch.ones_like(x, device=x.device) * 0.0

    # concat into xyz
    xyz = torch.stack([x, y, z], dim=-1)

    #
    # print(xy_pix.shape, "xy_pix", xy_pix.max(), xy_pix.min())
    cam_origin_world = xyz

    # Return tuple of cam_origins, ray_world_directions
    return cam_origin_world, rd_world_hom[..., :3]


def get_opencv_pixel_coordinates(y_resolution: int, x_resolution: int, device="cpu"):
    """For an image with y_resolution and x_resolution, return a tensor of pixel coordinates
    normalized to lie in [0, 1], with the origin (0, 0) in the top left corner,
    the x-axis pointing right, the y-axis pointing down, and the bottom right corner
    being at (1, 1).

    Returns:
        xy_pix: a meshgrid of values from [0, 1] of shape
                (y_resolution, x_resolution, 2)
    """
    i, j = torch.meshgrid(
        torch.linspace(0, 1, steps=x_resolution, device=device),
        torch.linspace(0, 1, steps=y_resolution, device=device),
        indexing="ij",
    )

    xy_pix = torch.stack([i.float(), j.float()], dim=-1).permute(1, 0, 2)
    return xy_pix


def project(xyz_cam_hom: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Projects homogenized 3D points xyz_cam_hom in camera coordinates
    to pixel coordinates.

    Args:
        xyz_cam_hom: 3D points of shape (..., 4)
        intrinsics: camera intrinscics of shape (..., 3, 3)

    Returns:
        xy: homogeneous pixel coordinates of shape (..., 3) (final coordinate is 1)
    """
    xyw = torch.einsum("...ij,...j->...i", intrinsics, xyz_cam_hom[..., :3])
    z = xyw[..., -1:]
    xyw = xyw / (z + 1e-9)  # z-divide
    return xyw[..., :3], z

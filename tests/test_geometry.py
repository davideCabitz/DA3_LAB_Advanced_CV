import torch

from da3.camera import affine_inverse, decode_pose, encode_pose, mat_to_quat, quat_to_mat
from da3.geometry import (
    make_ray_map,
    pixel_centers,
    points_from_depth_ray,
    pose_from_ray_map,
    rq_decomposition,
    unproject_depth,
)
from helpers import random_camera, rot_angle_deg

H, W = 240, 320


def test_rq_decomposition():
    K, c2w = random_camera(B=4)
    R = c2w[:, :3, :3].transpose(-1, -2)
    Kr, Rr = rq_decomposition(3.7 * K @ R)
    Kr = Kr / Kr[:, 2:3, 2:3]
    assert torch.allclose(Kr, K, atol=1e-8) and torch.allclose(Rr, R, atol=1e-10)


def test_ray_map_to_pose_round_trip():
    """Synthetic K, R, t -> ray map -> Eq. 1 (center) + Eq. 2 (DLT + RQ) recovers K, R, t."""
    K, c2w = random_camera(B=5, H=H, W=W)
    for ray_hw in [(H, W), (H // 2, W // 2), (137, 183)]:
        ray = make_ray_map(K, c2w, (H, W), ray_hw)
        w2c, K_hat = pose_from_ray_map(ray, (H, W))
        w2c_gt = affine_inverse(c2w)[:, :3]
        assert torch.allclose(K_hat, K, rtol=1e-6, atol=1e-5), (K_hat - K).abs().max()
        assert rot_angle_deg(w2c[:, :3, :3], w2c_gt[:, :3, :3]).max() < 1e-5
        assert torch.allclose(w2c[:, :, 3], w2c_gt[:, :, 3], atol=1e-8)


def test_ray_map_to_pose_with_noise():
    """Plain DLT (Eq. 2) on rays with small Gaussian noise stays within 0.1 deg / 0.5% focal."""
    K, c2w = random_camera(B=3, H=H, W=W, seed=1)
    ray = make_ray_map(K, c2w, (H, W), (H // 2, W // 2))
    g = torch.Generator().manual_seed(0)
    noisy = ray.clone()
    noisy[..., 3:] += 1e-3 * torch.randn(noisy[..., 3:].shape, generator=g, dtype=ray.dtype)
    noisy[..., :3] += 1e-3 * torch.randn(noisy[..., :3].shape, generator=g, dtype=ray.dtype)
    w2c, K_hat = pose_from_ray_map(noisy, (H, W))
    R_gt = c2w[:, :3, :3].transpose(-1, -2)
    assert rot_angle_deg(w2c[:, :3, :3], R_gt).max() < 0.1
    assert ((K_hat[:, 0, 0] - K[:, 0, 0]).abs() / K[:, 0, 0]).max() < 0.005


def test_ray_layout_is_origin_then_direction():
    K, c2w = random_camera(B=1, H=H, W=W)
    ray = make_ray_map(K, c2w, (H, W), (4, 5))
    assert torch.allclose(ray[0, 2, 3, :3], c2w[0, :3, 3])  # origin = camera centre
    cam = torch.linalg.inv(c2w[0, :3, :3]) @ ray[0, 2, 3, 3:]
    assert torch.isclose(cam[2], torch.tensor(1.0, dtype=cam.dtype))  # unnormalized, z_cam = 1


def test_depth_ray_points_consistency():
    """P = t + D·d equals R (D K^-1 p) + t and reprojects to the pixel with depth D."""
    K, c2w = random_camera(B=2, H=H, W=W)
    depth = 0.5 + 5 * torch.rand(2, H, W, dtype=torch.float64)
    P = points_from_depth_ray(depth, make_ray_map(K, c2w, (H, W)))
    p = pixel_centers(H, W, H, W, dtype=torch.float64)
    cam = depth[..., None] * torch.einsum("bij,hwj->bhwi", torch.linalg.inv(K), p)
    P_ref = torch.einsum("bij,bhwj->bhwi", c2w[:, :3, :3], cam) + c2w[:, None, None, :3, 3]
    assert torch.allclose(P, P_ref, atol=1e-10)
    assert torch.allclose(unproject_depth(depth, K, c2w), P_ref, atol=1e-10)
    # reprojection
    w2c = affine_inverse(c2w)
    Xc = torch.einsum("bij,bhwj->bhwi", w2c[:, :3, :3], P) + w2c[:, None, None, :3, 3]
    assert torch.allclose(Xc[..., 2], depth, atol=1e-9)
    uv = torch.einsum("bij,bhwj->bhwi", K, Xc / Xc[..., 2:3])
    assert torch.allclose(uv, p.expand_as(uv), atol=1e-8)


def test_quaternion_and_pose_encoding_round_trip():
    K, c2w = random_camera(B=6, H=H, W=W)
    R = c2w[:, :3, :3]
    q = mat_to_quat(R)
    assert torch.allclose(quat_to_mat(q), R, atol=1e-10) and (q[:, 3] >= 0).all()
    K[:, 0, 2], K[:, 1, 2] = W / 2, H / 2  # encoding assumes a centered principal point
    c2w_hat, K_hat = decode_pose(encode_pose(c2w, K, (H, W)).double(), (H, W))
    assert torch.allclose(c2w_hat, c2w[:, :3], atol=1e-5) and torch.allclose(K_hat, K, atol=1e-3)

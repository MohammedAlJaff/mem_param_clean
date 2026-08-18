"""
Synthetic data on a standard 3D torus embedded into ambient D dimensions.

The torus is parameterized by two intrinsic angles:
    theta: rotation around the central axis
    phi: rotation around the tube

The canonical donut lives in R3. The returned ambient tensor has shape [N, D]
after an orthonormal projection basis maps canonical torus coordinates into the
observed space. Geometry metadata includes that basis and the original 3D torus
coordinates so replay metrics can measure both copying and manifold drift.
"""

import math

import torch


def torus_angles_to_xyz(theta, phi, major_radius, minor_radius):
    """
    Map torus angles to ambient 3D coordinates.
    """
    radius = major_radius + minor_radius * torch.cos(phi)
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    z = minor_radius * torch.sin(phi)
    return torch.stack((x, y, z), dim=-1)


def torus_xyz_to_angles(x, major_radius, minor_radius):
    """
    Project canonical 3D points to nearest torus angles under the radial chart.

    This inverse map is used by metrics to express generated samples in the
    torus intrinsic coordinates.
    """
    xy_radius = torch.linalg.norm(x[..., :2], dim=-1)
    theta = torch.atan2(x[..., 1], x[..., 0])
    phi = torch.atan2(x[..., 2], xy_radius - float(major_radius))
    return theta, phi


def torus_periodic_features(theta, phi):
    """
    Represent periodic torus coordinates without angle-wrap discontinuities.

    Raw angles jump at the wrap point. The cosine/sine representation makes
    nearest-neighbor and sliced-W2 metrics continuous at that boundary.
    """
    return torch.stack(
        (
            torch.cos(theta),
            torch.sin(theta),
            torch.cos(phi),
            torch.sin(phi),
        ),
        dim=-1,
    )


def _make_basis(config):
    """
    Build the fixed ambient projection basis for the canonical 3D donut.
    """
    ambient_dim = int(config.data.ambient_dim)
    base_embedding_dim = 3
    if ambient_dim < base_embedding_dim:
        raise ValueError("torus3d requires ambient_dim >= 3.")

    generator = torch.Generator().manual_seed(int(config.data.geometry_seed))
    basis, _ = torch.linalg.qr(
        torch.randn(ambient_dim, base_embedding_dim, generator=generator)
    )
    return basis[:, :base_embedding_dim].contiguous()


def _sample_torus(config, basis, n_samples, seed):
    """
    Draw torus angles and return their canonical and ambient coordinates.
    """
    generator = torch.Generator().manual_seed(int(seed))
    theta = 2.0 * math.pi * torch.rand(n_samples, generator=generator)
    phi = 2.0 * math.pi * torch.rand(n_samples, generator=generator)
    major_radius = float(config.data.major_radius)
    minor_radius = float(config.data.minor_radius)
    torus_xyz = torus_angles_to_xyz(theta, phi, major_radius, minor_radius)
    ambient = torus_xyz @ basis.T
    sigma_eta = float(getattr(config.data, "sigma_eta", 0.0))
    if sigma_eta > 0.0:
        ambient = ambient + sigma_eta * torch.randn(
            ambient.shape,
            generator=generator,
            dtype=ambient.dtype,
        )
    angles = torch.stack((theta, phi), dim=-1)
    angle_features = torus_periodic_features(theta, phi)
    return (
        ambient.contiguous(),
        torus_xyz.contiguous(),
        angles.contiguous(),
        angle_features.contiguous(),
    )


def load_torus3d_bundle(config):
    """
    Build train/reference tensors for a controlled projected-3D-torus study.

    The saved geometry contains both ambient projection information and
    canonical torus coordinates. Synthetic evaluation uses that metadata to
    compare generated points in ambient space and in periodic intrinsic
    features.
    """
    basis = _make_basis(config)
    ambient_dim = int(config.data.ambient_dim)
    train, train_torus_xyz, train_angles, train_angle_features = _sample_torus(
        config,
        basis,
        int(config.data.n_train),
        int(config.training.seed),
    )
    reference, reference_torus_xyz, reference_angles, reference_angle_features = _sample_torus(
        config,
        basis,
        int(config.data.n_test),
        int(config.training.seed) + 1,
    )
    major_radius = torch.tensor(float(config.data.major_radius))
    minor_radius = torch.tensor(float(config.data.minor_radius))
    sigma_eta = torch.tensor(float(getattr(config.data, "sigma_eta", 0.0)))
    ambient_dim_tensor = torch.tensor(ambient_dim)

    return {
        "train": train,
        "train_labels": None,
        "reference": reference,
        "reference_labels": None,
        "train_role": "train",
        "reference_role": "heldout_reference",
        "train_source_id": f"torus3d:seed:{int(config.training.seed)}",
        "reference_source_id": f"torus3d:seed:{int(config.training.seed) + 1}",
        "train_split": "synthetic_train_seed",
        "reference_split": "synthetic_reference_seed",
        "geometry": {
            "kind": "torus3d",
            "ambient_dim": ambient_dim_tensor,
            "base_embedding_dim": torch.tensor(3),
            "local_manifold_dim": torch.tensor(2),
            "metric_feature_dim": torch.tensor(4),
            "basis": basis,
            "major_radius": major_radius,
            "minor_radius": minor_radius,
            "sigma_eta": sigma_eta,
            "train_torus_xyz": train_torus_xyz,
            "reference_torus_xyz": reference_torus_xyz,
            "train_angles": train_angles,
            "reference_angles": reference_angles,
            "train_intrinsic": train_angle_features,
            "reference_intrinsic": reference_angle_features,
            "train_angle_features": train_angle_features,
            "reference_angle_features": reference_angle_features,
        },
        "display_shape": None,
    }


if __name__ == "__main__":
    from mem_param_src.config import get_base_config

    print("mem_param_src.datasets.torus example")
    config = get_base_config()
    config.data.dataset = "torus3d"
    config.data.n_train = 8
    config.data.n_test = 6
    config.data.ambient_dim = 8
    config.data.data_dim = 8

    bundle = load_torus3d_bundle(config)
    print(f"train shape           : {tuple(bundle['train'].shape)}")
    print(f"reference shape       : {tuple(bundle['reference'].shape)}")
    print(f"basis shape           : {tuple(bundle['geometry']['basis'].shape)}")
    print(f"train torus xyz shape : {tuple(bundle['geometry']['train_torus_xyz'].shape)}")
    print(f"train angles shape    : {tuple(bundle['geometry']['train_angles'].shape)}")
    print(f"train intrinsic shape : {tuple(bundle['geometry']['train_intrinsic'].shape)}")

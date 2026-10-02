"""Image-to-vector feature extraction for memorization metrics."""

from pathlib import Path

import torch
import torch.nn.functional as F

from mem_param_src.metrics.image.preprocessing import (
    device_from_arg,
    normalize_images,
    select_cuda_device,
    to_uint8_rgb,
)


def _coerce_feature_output(output):
    """Convert an encoder output to a `[B, F]` feature matrix."""
    if isinstance(output, dict):
        if "features" in output:
            output = output["features"]
        elif len(output) == 1:
            output = next(iter(output.values()))
        else:
            raise ValueError(
                "Callable encoder returned a dict with multiple feature tensors."
            )
    elif isinstance(output, (tuple, list)):
        if len(output) != 1:
            raise ValueError("Callable encoder must return one feature tensor.")
        output = output[0]

    if not torch.is_tensor(output):
        raise TypeError("Encoder output must be a torch.Tensor.")
    return output.flatten(1).float()


def _encode_with_callable(images, encoder, *, batch_size, image_range, device):
    """Encode images with a callable while preserving its training mode."""
    images = normalize_images(images, image_range=image_range)
    torch_device = device_from_arg(device)

    was_training = getattr(encoder, "training", None)
    if hasattr(encoder, "eval"):
        encoder.eval()
    if hasattr(encoder, "to"):
        encoder.to(torch_device)

    features = []
    try:
        with torch.no_grad():
            for start in range(0, images.shape[0], batch_size):
                batch = images[start : start + batch_size].to(torch_device)
                encoded = _coerce_feature_output(encoder(batch))
                features.append(encoded.detach().cpu())
    finally:
        if was_training is not None and hasattr(encoder, "train"):
            encoder.train(was_training)

    return torch.cat(features, dim=0)


def _rgb_batch(images):
    """Return a float RGB batch without changing its value range."""
    channels = int(images.shape[1])
    if channels == 1:
        return images.repeat(1, 3, 1, 1)
    if channels == 3:
        return images
    if channels == 4:
        return images[:, :3]
    raise ValueError(f"Expected 1, 3, or 4 image channels, got {channels}")


class SSCDFeatureEncoder(torch.nn.Module):
    """Official SSCD TorchScript descriptor with fixed image preprocessing."""

    def __init__(self, model_path, input_size=320):
        super().__init__()
        model_path = Path(model_path).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing SSCD TorchScript model: {model_path}")
        input_size = int(input_size)
        if input_size < 1:
            raise ValueError(f"input_size must be at least 1, got {input_size}")

        self.model_path = str(model_path)
        self.input_size = input_size
        self.model = torch.jit.load(str(model_path), map_location="cpu")
        self.model.eval()
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1),
        )
        self.name = f"sscd:disc_mixup:final_512:{input_size}"
        self.feature_layer = "final_512_l2"

    def train(self, mode=True):
        """Keep the frozen descriptor in evaluation mode."""
        super().train(False)
        self.model.eval()
        return self

    def forward(self, images):
        """Return one L2-normalized 512-dimensional descriptor per image."""
        images = _rgb_batch(images)
        images = F.interpolate(
            images,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        images = (images - self.mean) / self.std
        features = _coerce_feature_output(self.model(images))
        return F.normalize(features, dim=1)


class LPIPSFeatureEncoder(torch.nn.Module):
    """Euclidean embedding whose squared distance equals LPIPS v0.1."""

    def __init__(
        self,
        input_size=32,
        *,
        network="alex",
        version="0.1",
        model=None,
        weights_path=None,
    ):
        super().__init__()
        input_size = int(input_size)
        if input_size < 1:
            raise ValueError(f"input_size must be at least 1, got {input_size}")
        if str(network) != "alex":
            raise ValueError("The validated LPIPS feature encoder requires network='alex'.")
        if str(version) != "0.1":
            raise ValueError("The validated LPIPS feature encoder requires version='0.1'.")

        if model is None:
            try:
                import lpips
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "lpips==0.1.4 is required for the LPIPS feature encoder."
                ) from exc
            model = lpips.LPIPS(
                pretrained=True,
                net="alex",
                version="0.1",
                lpips=True,
                spatial=False,
                eval_mode=True,
                verbose=False,
                model_path=(
                    None if weights_path is None else str(Path(weights_path))
                ),
            )

        self.input_size = input_size
        self.network = str(network)
        self.version = str(version)
        self.model = model
        self.weights_path = (
            None if weights_path is None else str(Path(weights_path).resolve())
        )
        self.model.eval()
        self.name = f"lpips:{self.network}:v{self.version}:sqrt:{input_size}"
        self.feature_layer = "relu1_relu5_weighted_normalized"

    def train(self, mode=True):
        """Keep the frozen perceptual network in evaluation mode."""
        super().train(False)
        self.model.eval()
        return self

    @staticmethod
    def _linear_weight(layer):
        convolution = next(
            module
            for module in layer.modules()
            if isinstance(module, torch.nn.Conv2d)
        )
        weight = convolution.weight[0, :, 0, 0]
        if bool((weight < -1.0e-8).any()):
            raise ValueError("LPIPS channel weights must be nonnegative.")
        return weight.clamp_min(0.0)

    def forward(self, images):
        """Return weighted normalized activations for exact LPIPS distances."""
        try:
            import lpips
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "lpips==0.1.4 is required for the LPIPS feature encoder."
            ) from exc

        images = _rgb_batch(images)
        if tuple(images.shape[-2:]) != (self.input_size, self.input_size):
            images = F.interpolate(
                images,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        images = images.mul(2.0).sub(1.0)
        scaled = self.model.scaling_layer(images)
        activations = self.model.net.forward(scaled)
        vectors = []
        for activation, linear in zip(activations, self.model.lins):
            normalized = lpips.normalize_tensor(activation)
            height, width = normalized.shape[-2:]
            weight = self._linear_weight(linear).to(normalized)
            scale = torch.sqrt(weight / float(height * width)).reshape(
                1, -1, 1, 1
            )
            vectors.append((normalized * scale).flatten(1))
        return torch.cat(vectors, dim=1).float()


class DINOv2FeatureEncoder(torch.nn.Module):
    """Pinned local DINOv2 ViT-S/14 final CLS descriptor."""

    def __init__(self, repo_path, weights_path, input_size=224):
        super().__init__()
        repo_path = Path(repo_path).expanduser().resolve()
        weights_path = Path(weights_path).expanduser().resolve()
        if not (repo_path / "hubconf.py").is_file():
            raise FileNotFoundError(f"Missing local DINOv2 source checkout: {repo_path}")
        if not weights_path.is_file():
            raise FileNotFoundError(f"Missing DINOv2 weights: {weights_path}")
        input_size = int(input_size)
        if input_size < 14 or input_size % 14:
            raise ValueError("DINOv2 input_size must be a positive multiple of 14.")

        self.repo_path = str(repo_path)
        self.weights_path = str(weights_path)
        self.input_size = input_size
        self.model = torch.hub.load(
            str(repo_path),
            "dinov2_vits14",
            source="local",
            pretrained=True,
            weights=str(weights_path),
            img_size=518,
        )
        self.model.eval()
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1),
        )
        self.name = f"dinov2:vits14:final_cls_l2:{input_size}"
        self.feature_layer = "final_cls_l2"

    def train(self, mode=True):
        """Keep the frozen DINOv2 backbone in evaluation mode."""
        super().train(False)
        self.model.eval()
        return self

    def forward(self, images):
        """Return one normalized 384-dimensional CLS descriptor per image."""
        images = _rgb_batch(images)
        images = F.interpolate(
            images,
            size=(self.input_size, self.input_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        images = (images - self.mean) / self.std
        features = _coerce_feature_output(self.model(images))
        return F.normalize(features, dim=1)


def parse_torch_fidelity_spec(spec):
    """Parse `torch_fidelity:<extractor>[:<layer>]`."""
    parts = str(spec).split(":")
    if len(parts) == 2 and parts[0] == "torch_fidelity":
        extractor_name = parts[1]
        try:
            from torch_fidelity.utils import resolve_feature_layer_for_metric
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "torch-fidelity is required for torch_fidelity:* image encoders."
            ) from exc
        feature_layer = resolve_feature_layer_for_metric(
            "fid",
            feature_extractor=extractor_name,
        )
        return extractor_name, feature_layer
    if len(parts) == 3 and parts[0] == "torch_fidelity":
        return parts[1], parts[2]
    raise ValueError(
        "Encoder spec must look like 'torch_fidelity:inception-v3-compat:2048'."
    )


class TorchFidelityFeatureEncoder:
    """Batched wrapper around one torch-fidelity feature extractor."""

    def __init__(self, spec, device, *, weights_path=None):
        self.spec = spec
        self.device = select_cuda_device(device)
        self.extractor_name, self.feature_layer = parse_torch_fidelity_spec(spec)
        self.weights_path = None if weights_path is None else str(Path(weights_path))

        try:
            from torch_fidelity.utils import create_feature_extractor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "torch-fidelity is required for torch_fidelity:* image encoders."
            ) from exc

        self.extractor = create_feature_extractor(
            self.extractor_name,
            [self.feature_layer],
            cuda=self.device.type == "cuda",
            verbose=False,
            **(
                {} if self.weights_path is None else {
                    "feature_extractor_weights_path": self.weights_path
                }
            ),
        )
        self.extractor.eval()

    def encode(self, batch):
        """Return one flat feature vector per image."""
        batch = to_uint8_rgb(batch, image_range="0_1").to(self.device)
        with torch.no_grad():
            output = self.extractor(batch)
        features = self.extractor.convert_features_tuple_to_dict(output)
        return features[self.feature_layer].flatten(1).float().cpu()


def _encode_with_torch_fidelity(images, encoder, *, batch_size, image_range):
    """Encode an image batch with a constructed torch-fidelity encoder."""
    images = normalize_images(images, image_range=image_range)
    features = []
    for start in range(0, images.shape[0], batch_size):
        features.append(encoder.encode(images[start : start + batch_size]))
    return torch.cat(features, dim=0)


def build_metric_vectors(
    samples,
    reference,
    *,
    encoder,
    batch_size,
    image_range,
    device,
):
    """Map two image batches into one shared nearest-neighbor vector space."""
    if encoder is None:
        torch_device = device_from_arg(device)
        sample_vectors = (
            normalize_images(samples, image_range=image_range)
            .flatten(1)
            .to(torch_device)
        )
        reference_vectors = (
            normalize_images(reference, image_range=image_range)
            .flatten(1)
            .to(torch_device)
        )
        return sample_vectors, reference_vectors, "pixels", "none"

    if isinstance(encoder, str):
        torch_fidelity_encoder = TorchFidelityFeatureEncoder(encoder, device)
        sample_vectors = _encode_with_torch_fidelity(
            samples,
            torch_fidelity_encoder,
            batch_size=batch_size,
            image_range=image_range,
        )
        reference_vectors = _encode_with_torch_fidelity(
            reference,
            torch_fidelity_encoder,
            batch_size=batch_size,
            image_range=image_range,
        )
        encoder_name = (
            f"torch_fidelity:{torch_fidelity_encoder.extractor_name}:"
            f"{torch_fidelity_encoder.feature_layer}"
        )
        return sample_vectors, reference_vectors, "features", encoder_name

    if not callable(encoder):
        raise TypeError(
            "encoder must be None, a callable, or a torch_fidelity:* spec string."
        )

    sample_vectors = _encode_with_callable(
        samples,
        encoder,
        batch_size=batch_size,
        image_range=image_range,
        device=device,
    )
    reference_vectors = _encode_with_callable(
        reference,
        encoder,
        batch_size=batch_size,
        image_range=image_range,
        device=device,
    )
    encoder_name = getattr(encoder, "name", encoder.__class__.__name__)
    return sample_vectors, reference_vectors, "features", str(encoder_name)


def encode_image_features(
    images,
    *,
    encoder,
    batch_size=256,
    image_range="auto",
    device="cpu",
):
    """Encode one image artifact into a reusable `[B, F]` feature matrix."""
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if encoder is None:
        return normalize_images(images, image_range=image_range).flatten(1).float()

    if isinstance(encoder, str):
        encoder = TorchFidelityFeatureEncoder(encoder, device)
    if isinstance(encoder, TorchFidelityFeatureEncoder):
        return _encode_with_torch_fidelity(
            images,
            encoder,
            batch_size=batch_size,
            image_range=image_range,
        )
    if not callable(encoder):
        raise TypeError(
            "encoder must be None, a callable, or a torch_fidelity:* spec string."
        )
    return _encode_with_callable(
        images,
        encoder,
        batch_size=batch_size,
        image_range=image_range,
        device=device,
    )


__all__ = [
    "DINOv2FeatureEncoder",
    "LPIPSFeatureEncoder",
    "SSCDFeatureEncoder",
    "TorchFidelityFeatureEncoder",
    "build_metric_vectors",
    "encode_image_features",
]

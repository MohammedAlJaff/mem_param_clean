"""Image-to-vector feature extraction for memorization metrics."""

import torch

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

    def __init__(self, spec, device):
        self.spec = spec
        self.device = select_cuda_device(device)
        self.extractor_name, self.feature_layer = parse_torch_fidelity_spec(spec)

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


__all__ = ["TorchFidelityFeatureEncoder", "build_metric_vectors"]

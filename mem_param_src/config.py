"""Base configuration helpers for first-party studies."""

import importlib.util
import json
import os
from collections.abc import Mapping
from pathlib import Path

from ml_collections import ConfigDict


def get_base_config():
    """
    Build the base nested config used by the clean source.
    """
    config = ConfigDict()

    study = config.study = ConfigDict()
    study.name = "unnamed_study"
    study.description = "Target parameterization and memorization dynamics."
    study.dataset_key = ""

    framework = config.framework = ConfigDict()
    framework.family = "fm"
    framework.prediction_space = "x"
    framework.loss_space = "x"
    framework.loss_weighting = "unit"

    network = config.network = ConfigDict()
    network.arch = "mlp"
    network.hidden_dim = 128
    network.n_layers = 3
    network.time_emb_dim = 64
    network.base_channels = 64
    network.channel_mult = (1, 2, 4, 4)
    network.attn_resolutions = (8,)
    network.num_res_blocks = 2
    network.dropout = 0.0

    data = config.data = ConfigDict()
    data.dataset = "gmm_subspace"
    data.root = "data"
    data.dataset_subdir = ""
    data.download = False
    data.train_split = "train"
    data.reference_split = "valid"
    data.resize_size = 0
    data.n_train = 256
    data.n_test = 256
    data.ambient_dim = 256
    data.intrinsic_dim = 4
    data.data_dim = 256
    data.image_size = 0
    data.image_channels = 0
    data.flatten = False
    data.balance_train_classes = False
    data.balance_reference_classes = False
    data.allow_same_source_reference = False
    data.n_clusters = 10
    data.sigma_z = 0.5
    data.sigma_eta = 1.0e-3
    data.geometry_seed = 0
    data.major_radius = 2.0
    data.minor_radius = 0.5
    data.recipe = "default"

    training = config.training = ConfigDict()
    training.seed = 0
    training.n_training_steps = 200
    training.batch_cap = 64
    training.batch_size = 64
    training.lr = 1.0e-3
    training.weight_decay = 0.0
    training.optimizer = "adam"
    training.momentum = 0.9
    training.ckpt_every = 50
    training.ckpt_schedule = "interval"
    training.ckpt_steps = ()
    training.resolved_ckpt_steps = ()
    training.log_every = 10
    training.show_progress = True
    training.start_time = 0.01
    training.end_time = 0.99
    training.use_safe_division = True
    training.safe_div_denom_eps = 0.01

    evaluation = config.evaluation = ConfigDict()
    evaluation.n_samples = 64
    evaluation.save_generated_samples = False
    evaluation.sample_bank_size = 1024
    evaluation.batch_size = 64
    evaluation.f_mem_threshold = 0.3
    evaluation.sample_panel_count = 8
    evaluation.chunk_size = 256
    evaluation.loss_eval_times = (0.5, 0.75)
    evaluation.loss_eval_batch_size = 0
    evaluation.sliced_w2_n_directions = 128

    sampling = config.sampling = ConfigDict()
    sampling.method = "heun"
    sampling.n_steps = 50
    sampling.start_time = 0.01
    sampling.end_time = 0.99
    sampling.use_safe_division = True
    sampling.safe_div_denom_eps = 0.01
    sampling.output = "x_1"

    outputs = config.outputs = ConfigDict()
    outputs.run_name = "run"

    monitoring = config.monitoring = ConfigDict()
    monitoring.batch_size = 64

    generation_monitor = config.generation_monitor = ConfigDict()
    generation_monitor.enabled = False
    generation_monitor.n_samples = 64
    generation_monitor.n_train_samples = 0
    generation_monitor.show = True

    random_projected_f_mem = config.random_projected_f_mem = ConfigDict()
    random_projected_f_mem.enabled = False
    random_projected_f_mem.n_samples = 512
    random_projected_f_mem.resize_size = 16
    random_projected_f_mem.projection_dim = 256
    random_projected_f_mem.projection_seed = None

    runtime = config.runtime = ConfigDict()
    runtime.precision = "fp32"
    runtime.tf32_enabled = False
    runtime.autocast_dtype = "none"
    runtime.pin_memory = True
    runtime.non_blocking_h2d = False

    tracking = config.tracking = ConfigDict()
    tracking.console = ConfigDict()
    tracking.console.enabled = True
    tracking.console.summary_every_steps = int(training.log_every)
    tracking.console.summary_every_seconds = 60.0
    tracking.console.stage_banners = True
    tracking.console.tee_to_attempt_log = True

    tracking.wandb = ConfigDict()
    tracking.wandb.enabled = False
    tracking.wandb.mode = "offline"
    tracking.wandb.project = ""
    tracking.wandb.name = ""
    tracking.wandb.group = ""
    tracking.wandb.entity = _default_wandb_entity("maj90")
    tracking.wandb.dir = "logs/wandb"
    tracking.wandb.tags = []
    tracking.wandb.notes = ""
    tracking.wandb.strict = False

    return config


def _default_wandb_entity(fallback):
    """Return the environment W&B entity or a fallback."""
    return os.getenv("WANDB_ENTITY", "").strip() or fallback


def config_to_dict(value):
    """Recursively convert config containers into plain Python types."""
    if isinstance(value, ConfigDict):
        return {key: config_to_dict(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {key: config_to_dict(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [config_to_dict(item) for item in value]
    if isinstance(value, list):
        return [config_to_dict(item) for item in value]
    return value


def clone_config(config):
    """Deep-clone a config through plain Python containers."""
    return ConfigDict(config_to_dict(config))


def _is_config_mapping(value):
    """Return whether a value is a ConfigDict or standard mapping."""
    return isinstance(value, (ConfigDict, Mapping))


def _set_dotted(config, dotted_key, value):
    """Set one existing leaf field using a dotted key."""
    parts = dotted_key.split(".")
    if not dotted_key or any(not part for part in parts):
        raise ValueError(f"Invalid configuration field: {dotted_key!r}")

    cursor = config
    for part in parts[:-1]:
        if part not in cursor:
            raise ValueError(f"Unknown configuration field: {dotted_key!r}")
        cursor = cursor[part]
        if not _is_config_mapping(cursor):
            raise ValueError(f"Configuration field is not a section: {part!r}")

    leaf = parts[-1]
    if leaf not in cursor:
        raise ValueError(f"Unknown configuration field: {dotted_key!r}")
    if _is_config_mapping(cursor[leaf]):
        raise ValueError(
            f"Configuration override must name a leaf field: {dotted_key!r}"
        )

    try:
        cursor[leaf] = value
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid value for configuration field {dotted_key!r}: {value!r}"
        ) from exc


def apply_overrides(config, overrides):
    """Apply strict dotted-key overrides onto an existing config in place."""
    for dotted_key, value in overrides.items():
        _set_dotted(config, dotted_key, value)
    return config


def parse_cli_overrides(items, allowed_sections=None):
    """Parse repeatable ``KEY=VALUE`` arguments into strict config overrides."""
    overrides = {}
    allowed_sections = set(allowed_sections or ())

    for item in items or ():
        if "=" not in item:
            raise ValueError(
                f"Configuration override must use KEY=VALUE syntax: {item!r}"
            )
        dotted_key, raw_value = item.split("=", 1)
        dotted_key = dotted_key.strip()
        if not dotted_key:
            raise ValueError(f"Configuration override has an empty key: {item!r}")
        if allowed_sections and dotted_key.split(".", 1)[0] not in allowed_sections:
            allowed = ", ".join(sorted(allowed_sections))
            raise ValueError(
                f"Override {dotted_key!r} is not allowed here. "
                f"Expected a field under: {allowed}."
            )

        raw_value = raw_value.strip()
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        overrides[dotted_key] = value

    return overrides


def _config_leaves(value, prefix=""):
    """Yield dotted leaf fields from a nested config or mapping."""
    for key, item in value.items():
        dotted_key = f"{prefix}.{key}" if prefix else str(key)
        if _is_config_mapping(item):
            yield from _config_leaves(item, dotted_key)
        else:
            yield dotted_key, item


def apply_config_overlay(config, overlay, allowed_sections=None):
    """Apply a partial config after checking its allowed top-level sections."""
    allowed_sections = set(allowed_sections or ())
    if allowed_sections:
        unexpected = sorted(set(overlay.keys()) - allowed_sections)
        if unexpected:
            allowed = ", ".join(sorted(allowed_sections))
            raise ValueError(
                "Evaluation config contains forbidden sections: "
                f"{', '.join(unexpected)}. Allowed sections: {allowed}."
            )

    return apply_overrides(config, dict(_config_leaves(overlay)))


def load_python_config(path, allowed_sections=None):
    """Load a Python config file through its required ``get_config`` function."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")

    module_name = f"_mem_param_config_{path.stem}_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import configuration file: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    get_config = getattr(module, "get_config", None)
    if not callable(get_config):
        raise ValueError(f"Configuration file must define get_config(): {path}")

    config = get_config()
    if isinstance(config, ConfigDict):
        loaded = config
    elif isinstance(config, Mapping):
        loaded = ConfigDict(config)
    else:
        raise TypeError(
            "get_config() must return a ConfigDict or mapping, "
            f"got {type(config).__name__}"
        )

    if allowed_sections:
        unexpected = sorted(set(loaded.keys()) - set(allowed_sections))
        if unexpected:
            allowed = ", ".join(sorted(allowed_sections))
            raise ValueError(
                "Evaluation config contains forbidden sections: "
                f"{', '.join(unexpected)}. Allowed sections: {allowed}."
            )
    return loaded


__all__ = [
    "apply_config_overlay",
    "apply_overrides",
    "clone_config",
    "config_to_dict",
    "get_base_config",
    "load_python_config",
    "parse_cli_overrides",
]

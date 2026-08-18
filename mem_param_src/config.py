"""Base configuration helpers for first-party studies."""

import os

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
    training.denominator_epsilon = 0.01

    evaluation = config.evaluation = ConfigDict()
    evaluation.n_samples = 64
    evaluation.sample_bank_size = 1024
    evaluation.batch_size = 64
    evaluation.ratio_threshold = 0.5
    evaluation.sample_panel_count = 8
    evaluation.chunk_size = 256
    evaluation.loss_eval_times = (0.5, 0.75)
    evaluation.loss_eval_batch_size = 0

    sampling = config.sampling = ConfigDict()
    sampling.method = "heun"
    sampling.n_steps = 50
    sampling.start_time = 0.01
    sampling.end_time = 0.99
    sampling.use_safe_division = True
    sampling.denominator_epsilon = 0.01
    sampling.output = "final_sample"

    outputs = config.outputs = ConfigDict()
    outputs.run_name = "run"

    monitoring = config.monitoring = ConfigDict()
    monitoring.batch_size = 64

    generation_monitor = config.generation_monitor = ConfigDict()
    generation_monitor.enabled = False
    generation_monitor.n_samples = 64
    generation_monitor.show = True

    random_projected_f_mem = config.random_projected_f_mem = ConfigDict()
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


def _set_dotted(config, dotted_key, value):
    """Set one nested config field using a dotted key."""
    parts = dotted_key.split(".")
    cursor = config
    for part in parts[:-1]:
        if part not in cursor:
            cursor[part] = ConfigDict()
        cursor = cursor[part]
    cursor[parts[-1]] = value


def apply_overrides(config, overrides):
    """Apply dotted-key overrides onto an existing config in place."""
    for dotted_key, value in overrides.items():
        if value is None:
            continue
        _set_dotted(config, dotted_key, value)
    return config


__all__ = [
    "apply_overrides",
    "clone_config",
    "config_to_dict",
    "get_base_config",
]

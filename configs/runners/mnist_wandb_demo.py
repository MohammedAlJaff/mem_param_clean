"""One MNIST run for observing training loss and fixed-noise samples."""

from mem_param_src.config import get_base_config


def get_config():
    """Return the proposed 10,000-update online monitoring demonstration."""
    config = get_base_config()
    config.study.name = "mnist_wandb_demo"
    config.study.dataset_key = "mnist"
    config.outputs.run_name = "mnist_wandb_demo"
    config.framework.prediction_space = "eps"
    config.framework.loss_space = "eps"
    config.data.dataset = "mnist"
    config.data.root = "data"
    config.data.download = False
    config.data.train_split = "train"
    config.data.reference_split = "test"
    config.data.n_train = 8
    config.data.n_test = 16
    config.data.image_size = 32
    config.data.resize_size = 32
    config.data.image_channels = 1
    config.data.flatten = False

    # Increase capacity with three levels and two residual blocks per level.
    config.network.arch = "unet"
    config.network.base_channels = 64
    config.network.channel_mult = (1, 2, 3)
    # Disable optional attention; the backbone still has middle-block attention.
    config.network.attn_resolutions = ()
    config.network.num_res_blocks = 2
    config.network.dropout = 0.0
    config.training.seed = 7
    config.training.lr = 1.0e-3
    config.training.batch_cap = 8
    config.training.n_training_steps = 10000
    config.training.ckpt_every = 250
    config.training.log_every = 25
    config.training.use_safe_division = False

    # Reuse the same 16 noise tensors at every checkpoint.
    config.evaluation.sample_bank_size = 16
    config.evaluation.n_samples = 16
    config.generation_monitor.enabled = True
    config.generation_monitor.n_samples = 16
    config.monitoring.batch_size = 16
    config.sampling.n_steps = 50
    config.sampling.use_safe_division = False

    # Fail if monitoring fails, since online monitoring is the purpose of this run.
    config.tracking.wandb.enabled = True
    config.tracking.wandb.mode = "online"
    config.tracking.wandb.entity = "maj90"
    config.tracking.wandb.project = "mnist-unet"
    config.tracking.wandb.strict = True
    return config

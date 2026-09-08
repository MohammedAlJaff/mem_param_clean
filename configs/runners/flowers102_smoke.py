"""Short Flowers102 run that verifies image training and checkpoint replay."""

from mem_param_src.config import get_base_config


def get_config():
    """Return a small Flowers102 runner configuration."""
    config = get_base_config()
    config.study.name = "flowers102_runner_smoke"
    config.study.dataset_key = "flowers102"
    config.outputs.run_name = "flowers102_runner_smoke"

    config.framework.prediction_space = "v"
    config.framework.loss_space = "v"

    config.data.dataset = "flowers102"
    config.data.root = "data"
    config.data.download = False
    config.data.train_split = "train"
    config.data.reference_split = "val"
    config.data.n_train = 4
    config.data.n_test = 16
    config.data.image_size = 32
    config.data.resize_size = 32
    config.data.image_channels = 3
    config.data.flatten = False

    config.network.arch = "unet"
    config.network.base_channels = 32
    config.network.channel_mult = (1, 2)
    config.network.attn_resolutions = ()
    config.network.num_res_blocks = 1
    config.network.dropout = 0.0

    config.training.seed = 7
    config.training.n_training_steps = 2
    config.training.batch_cap = 4
    config.training.lr = 1.0e-3
    config.training.ckpt_every = 1
    config.training.log_every = 1

    config.evaluation.n_samples = 4
    config.evaluation.sample_bank_size = 4
    config.evaluation.batch_size = 2
    config.evaluation.chunk_size = 4

    # Reuse all four saved noise tensors for checkpoint previews.
    config.generation_monitor.n_samples = 4
    config.monitoring.batch_size = 2

    config.sampling.n_steps = 2
    config.sampling.output = "x_1"
    return config

"""CelebA32 eps/eps with 32 faces and saved checkpoint monitoring samples."""

from mem_param_src.config import get_base_config


def get_config():
    """Return a grayscale CelebA32 diagnostic config with exact sampling division."""
    config = get_base_config()
    config.study.name = "celeba32_eps_eps_monitor"
    config.study.dataset_key = "celeba32"
    config.outputs.run_name = "celeba32_eps_eps_n32_1k"
    config.framework.prediction_space = "eps"
    config.framework.loss_space = "eps"
    config.framework.loss_weighting = "unit"

    # Preserve the historical aligned-face crop and grayscale32 representation.
    config.data.dataset = "celeba32_grey"
    config.data.root = "data"
    config.data.download = False
    config.data.train_split = "train"
    config.data.reference_split = "valid"
    config.data.n_train = 32
    config.data.n_test = 500
    config.data.image_size = 32
    config.data.resize_size = 32
    config.data.image_channels = 1
    config.data.flatten = False

    # Use the current UNet; middle attention remains active.
    config.network.arch = "unet"
    config.network.base_channels = 64
    config.network.channel_mult = (1, 2, 3)
    config.network.attn_resolutions = ()
    config.network.num_res_blocks = 2
    config.network.dropout = 0.0
    config.training.seed = 7
    config.training.lr = 1.0e-3
    config.training.batch_cap = 32
    config.training.n_training_steps = 1000
    config.training.ckpt_every = 250
    config.training.log_every = 25
    config.training.use_safe_div = False

    # Save all 500 samples; show 64 below the same 32 training images.
    config.evaluation.sample_bank_size = 500
    config.evaluation.n_samples = 500
    config.evaluation.save_generated_samples = True
    config.generation_monitor.enabled = True
    config.generation_monitor.n_samples = 64
    config.generation_monitor.n_train_samples = 32
    config.random_projected_f_mem.enabled = True
    config.random_projected_f_mem.n_samples = 500
    config.monitoring.batch_size = 32
    config.sampling.method = "heun"
    config.sampling.n_steps = 50
    config.sampling.use_safe_div = False
    config.tracking.wandb.enabled = False
    return config

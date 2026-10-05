"""CelebA32 N=16 50k endpoint-range ablation with no self-attention."""

from mem_param_src.config import get_base_config


def get_config():
    """Build the shared configuration for the labeled time-range comparison."""
    config = get_base_config()
    config.study.name = "celeba32_n16_50k_time_range_noattn"
    config.study.dataset_key = "celeba32"
    config.outputs.run_name = "celeba32_n16_50k_time_range_noattn"
    config.framework.prediction_space = "eps"
    config.framework.loss_space = "eps"
    config.framework.loss_weighting = "unit"

    config.data.dataset = "celeba32_grey"
    config.data.root = "../old_mem_par/data"
    config.data.download = False
    config.data.train_split = "train"
    config.data.reference_split = "valid"
    config.data.n_train = 16
    config.data.n_test = 1024
    config.data.image_size = 32
    config.data.resize_size = 32
    config.data.image_channels = 1
    config.data.flatten = False

    # Keep the previous base32_m124 capacity, but remove all self-attention:
    # no resolution attention and no middle-block attention.
    config.network.arch = "unet"
    config.network.base_channels = 32
    config.network.channel_mult = (1, 2, 4)
    config.network.time_emb_dim = 128
    config.network.attn_resolutions = ()
    config.network.middle_attention = False
    config.network.num_res_blocks = 2
    config.network.dropout = 0.1

    config.training.seed = 0
    config.training.optimizer = "adam"
    config.training.lr = 5.0e-4
    config.training.weight_decay = 0.0
    config.training.batch_cap = 16
    config.training.n_training_steps = 50_000
    config.training.ckpt_schedule = "interval"
    config.training.ckpt_every = 1_000
    config.training.ckpt_steps = ()
    config.training.log_every = 25
    # The array supplies each row's training and sampling endpoints.
    config.training.start_time = 0.001
    config.training.end_time = 0.999
    config.training.use_safe_div = False
    config.training.safe_div_denominator_floor = 0.05

    config.sampling.method = "heun"
    config.sampling.n_steps = 50
    config.sampling.start_time = 0.001
    config.sampling.end_time = 0.999
    config.sampling.use_safe_div = True
    config.sampling.safe_div_denominator_floor = 0.05
    config.sampling.eps_to_v = "direct"
    config.sampling.heun_final_euler = True
    config.sampling.final_pixel_clamp = True
    config.sampling.output = "x_t_max"

    config.evaluation.sample_bank_size = 256
    config.evaluation.n_samples = 256
    config.evaluation.batch_size = 64
    config.evaluation.save_generated_samples = True
    config.generation_monitor.enabled = True
    config.generation_monitor.show = False
    config.generation_monitor.n_samples = 64
    config.generation_monitor.n_train_samples = 16
    config.random_projected_f_mem.enabled = True
    config.random_projected_f_mem.n_samples = 256
    config.random_projected_f_mem.resize_size = 16
    config.random_projected_f_mem.projection_dim = 256
    config.random_projected_f_mem.projection_seed = 3001
    config.monitoring.batch_size = 64

    config.runtime.precision = "fp32"
    config.runtime.tf32_enabled = True
    config.runtime.autocast_dtype = "none"
    config.runtime.non_blocking_h2d = True

    config.tracking.wandb.enabled = True
    config.tracking.wandb.mode = "online"
    config.tracking.wandb.strict = True
    config.tracking.wandb.tags = [
        "celeba32",
        "n16",
        "steps-50000",
        "time-range",
        "no-attention",
    ]
    config.tracking.wandb.notes = (
        "N=16 endpoint-range ablation: Adam lr=5e-4, Heun-50/final-Euler, "
        "50k updates, 1k checkpoints, and no UNet self-attention."
    )
    return config

"""Fresh CelebA32 U-Net-S native-v/v loss-weighting control settings."""

from mem_param_src.config import get_base_config


def get_config():
    """Return the shared one-million-update CelebA32 configuration."""
    config = get_base_config()
    config.study.name = "celeba32_unet_vv_weighting_1m"
    config.study.dataset_key = "celeba32-gray"
    config.outputs.run_name = "celeba32_unet_vv_weighting_1m"

    config.framework.prediction_space = "v"
    config.framework.loss_space = "v"
    config.framework.loss_weighting = "unit"

    config.data.dataset = "celeba32_grey"
    config.data.root = "../old_mem_par/data"
    config.data.download = False
    config.data.train_split = "train"
    config.data.reference_split = "valid"
    config.data.n_train = 512
    config.data.n_test = 1024
    config.data.image_size = 32
    config.data.resize_size = 32
    config.data.image_channels = 1
    config.data.flatten = False
    config.data.balance_train_classes = False
    config.data.balance_reference_classes = False

    config.network.arch = "unet"
    config.network.base_channels = 32
    config.network.channel_mult = (1, 2, 4)
    config.network.num_res_blocks = 2
    config.network.dropout = 0.1
    config.network.time_emb_dim = 128
    config.network.attn_resolutions = (16, 8)
    config.network.middle_attention = True
    config.network.learn_sigma = False

    config.training.seed = 0
    config.training.optimizer = "adam"
    config.training.lr = 1.0e-4
    config.training.weight_decay = 0.0
    config.training.batch_cap = 256
    config.training.effective_batch_size = 256
    config.training.n_training_steps = 1_000_000
    config.training.ckpt_schedule = "bonnaire-1m"
    config.training.ckpt_steps = ()
    config.training.recovery_every = 5_000
    config.training.log_every = 250
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
    config.sampling.eps_to_v = "via_x"
    config.sampling.heun_final_euler = False
    config.sampling.output = "x_t_max"
    config.sampling.final_pixel_clamp = False

    config.evaluation.sample_bank_size = 128
    config.evaluation.n_samples = 128
    config.evaluation.batch_size = 64
    config.evaluation.save_generated_samples = True
    config.generation_monitor.enabled = True
    config.generation_monitor.show = False
    config.generation_monitor.n_samples = 64
    config.generation_monitor.n_train_samples = 32
    config.random_projected_f_mem.enabled = True
    config.random_projected_f_mem.n_samples = 128
    config.random_projected_f_mem.resize_size = 16
    config.random_projected_f_mem.projection_dim = 256
    config.random_projected_f_mem.projection_seed = 3001
    config.monitoring.batch_size = 64

    config.runtime.precision = "fp32"
    config.runtime.tf32_enabled = False
    config.runtime.deterministic = True
    config.runtime.autocast_dtype = "none"
    config.runtime.non_blocking_h2d = True

    config.tracking.wandb.enabled = True
    config.tracking.wandb.mode = "online"
    config.tracking.wandb.strict = True
    config.tracking.wandb.project = "celeba32-unet-vv-weighting-1m"
    config.tracking.wandb.subexperiment = "celeba32-unet-vv-weighting-1m"
    config.tracking.wandb.tags = [
        "celeba32",
        "unet-s",
        "native-vv",
        "loss-weighting-control",
        "one-million",
        "via-x",
        "fresh-runs",
    ]
    config.tracking.wandb.notes = (
        "Fresh CelebA32 U-Net-S native-v/v loss-weighting control; "
        "N=512; seeds={0,1,2}; weights={x_equivalent,unit,eps_equivalent}; "
        "one million updates."
    )
    return config

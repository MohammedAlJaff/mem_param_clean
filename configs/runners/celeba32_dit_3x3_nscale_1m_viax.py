"""Production CelebA32-gray Small-DiT 3x3 and training-size matrix."""

from mem_param_src.config import get_base_config


def get_config():
    """Return the shared one-million-update production configuration."""
    config = get_base_config()
    config.study.name = "celeba32_dit_3x3_nscale_1m_viax"
    config.study.dataset_key = "celeba32-gray"
    config.outputs.run_name = "celeba32_dit_3x3_nscale_1m_viax"

    # The array manifest overrides these fields for each matrix row.
    config.framework.prediction_space = "x"
    config.framework.loss_space = "x"
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

    config.network.arch = "dit"
    config.network.depth = 12
    config.network.hidden_size = 384
    config.network.num_heads = 6
    config.network.mlp_ratio = 4.0
    config.network.patch_size = 2
    config.network.learn_sigma = False

    config.training.seed = 0
    config.training.optimizer = "adam"
    config.training.lr = 1.0e-4
    config.training.weight_decay = 0.0
    config.training.batch_cap = 64
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
    config.runtime.tf32_enabled = True
    config.runtime.deterministic = True
    config.runtime.autocast_dtype = "none"
    config.runtime.non_blocking_h2d = True

    config.tracking.wandb.enabled = True
    config.tracking.wandb.mode = "online"
    # Fail the allocation if W&B cannot preserve the configured run identity.
    # Exact local recovery remains available for a manual resume.
    config.tracking.wandb.strict = True
    config.tracking.wandb.subexperiment = "dit-3x3-nscale-1m-viax"
    config.tracking.wandb.tags = [
        "celeba32",
        "small-dit",
        "prediction-loss-3x3",
        "n-scale",
        "1m",
        "via-x",
        "safe-sampling",
        "seed-0",
    ]
    config.tracking.wandb.notes = (
        "Small DiT-S/2 full prediction/loss 3x3 across "
        "N={512,1024,2048,4096}; seed 0; one million updates. "
        "Sampling uses via-x conversion and safe division with floor 0.05."
    )
    return config

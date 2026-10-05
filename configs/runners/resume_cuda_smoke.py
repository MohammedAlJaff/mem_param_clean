"""Small CUDA run for interrupted-versus-uninterrupted state comparison."""

from mem_param_src.config import get_base_config


def get_config():
    """Return a short MNIST UNet run with accumulation and dropout."""
    config = get_base_config()
    config.study.name = "resume_cuda_smoke"
    config.study.dataset_key = "mnist16"
    config.outputs.run_name = "resume_cuda_smoke"
    config.framework.prediction_space = "eps"
    config.framework.loss_space = "eps"

    config.data.dataset = "mnist"
    config.data.root = "../old_mem_par/data"
    config.data.download = False
    config.data.train_split = "train"
    config.data.reference_split = "test"
    config.data.n_train = 16
    config.data.n_test = 16
    config.data.image_size = 16
    config.data.resize_size = 16
    config.data.image_channels = 1

    config.network.arch = "unet"
    config.network.base_channels = 8
    config.network.channel_mult = (1, 2)
    config.network.time_emb_dim = 16
    config.network.attn_resolutions = ()
    config.network.middle_attention = False
    config.network.num_res_blocks = 1
    config.network.dropout = 0.1

    config.training.seed = 123
    config.training.optimizer = "adam"
    config.training.lr = 1.0e-4
    config.training.batch_cap = 8
    config.training.effective_batch_size = 16
    config.training.n_training_steps = 40
    config.training.ckpt_schedule = "interval"
    config.training.ckpt_every = 10
    config.training.recovery_every = 5
    config.training.log_every = 5
    config.training.start_time = 0.001
    config.training.end_time = 0.999
    config.training.use_safe_div = False

    config.evaluation.n_samples = 8
    config.evaluation.sample_bank_size = 8
    config.evaluation.batch_size = 8
    config.generation_monitor.enabled = False
    config.random_projected_f_mem.enabled = False

    config.runtime.tf32_enabled = True
    config.runtime.deterministic = True
    config.runtime.non_blocking_h2d = True
    config.tracking.wandb.enabled = False
    return config

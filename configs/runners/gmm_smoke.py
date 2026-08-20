"""Short GMM run that verifies training and saved checkpoint replay."""

from mem_param_src.config import get_base_config


def get_config():
    """Return a small CPU-compatible GMM runner configuration."""
    config = get_base_config()
    config.study.name = "gmm_runner_smoke"
    config.study.dataset_key = "gmm_subspace"
    config.outputs.run_name = "gmm_runner_smoke"

    config.framework.prediction_space = "v"
    config.framework.loss_space = "v"

    config.data.dataset = "gmm_subspace"
    config.data.n_train = 32
    config.data.n_test = 32
    config.data.data_dim = 8
    config.data.ambient_dim = 8
    config.data.intrinsic_dim = 2
    config.data.n_clusters = 3

    config.network.arch = "mlp"
    config.network.hidden_dim = 32
    config.network.n_layers = 2
    config.network.time_emb_dim = 16

    config.training.n_training_steps = 4
    config.training.batch_cap = 16
    config.training.ckpt_every = 2
    config.training.log_every = 1

    config.evaluation.n_samples = 16
    config.evaluation.sample_bank_size = 16
    config.evaluation.batch_size = 8
    config.evaluation.chunk_size = 16
    config.evaluation.sliced_w2_n_directions = 16

    config.sampling.n_steps = 4
    config.sampling.output = "x_1"
    return config

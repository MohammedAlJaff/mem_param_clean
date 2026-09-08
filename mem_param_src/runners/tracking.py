"""Minimal W&B logging for image training runs."""

import importlib
import re
from pathlib import Path

from mem_param_src.config import config_to_dict
from mem_param_src.runners.artifacts import update_run_record


def _slug(value, fallback):
    """Return one compact W&B name component."""
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text).strip("-")
    return text or fallback


def resolve_wandb_identity(config):
    """Resolve project, run name, and comparison group from the study config."""
    dataset = _slug(config.study.dataset_key or config.data.dataset, "dataset")
    model = _slug(config.network.arch, "model")
    prediction = _slug(config.framework.prediction_space, "prediction")
    loss = _slug(config.framework.loss_space, "loss")
    n_train = int(config.data.n_train)
    batch_cap = int(config.training.batch_cap)
    n_steps = int(config.training.n_training_steps)
    seed = int(config.training.seed)

    defaults = {
        "project": f"{dataset}-{model}",
        "name": (
            f"{dataset}-{model}-N{n_train}-B{batch_cap}-"
            f"{prediction}_{loss}-steps{n_steps}-s{seed}"
        ),
        "group": f"{dataset}-{model}-steps{n_steps}",
    }
    wandb_config = config.tracking.wandb
    for field, default in defaults.items():
        value = str(getattr(wandb_config, field, "") or "").strip()
        setattr(wandb_config, field, value or default)
    return {field: str(getattr(wandb_config, field)) for field in defaults}


def _wandb_failure(state, exc, context):
    """Disable failed tracking without changing the model update sequence."""
    message = f"W&B {context} failed: {type(exc).__name__}: {exc}"
    run_dir = state["run_dir"]
    update_run_record(
        run_dir,
        wandb={
            **state["metadata"],
            "active": False,
            "warning": message,
        },
    )
    if state["run"] is not None:
        try:
            state["run"].finish()
        except Exception:
            pass
        state["run"] = None
    if state["strict"]:
        update_run_record(run_dir, state="failed", error=message)
        raise exc
    print(f"warning: {message}; training continues without W&B", flush=True)
    return None


def start_wandb(config, run_dir):
    """Start one optional W&B run and save its local identity."""
    if not bool(config.tracking.wandb.enabled):
        return None

    identity = resolve_wandb_identity(config)
    mode = str(config.tracking.wandb.mode)
    if mode not in ("online", "offline"):
        raise ValueError("tracking.wandb.mode must be 'online' or 'offline'.")

    relative_dir = Path(str(config.tracking.wandb.dir))
    if relative_dir.is_absolute() or ".." in relative_dir.parts:
        raise ValueError("tracking.wandb.dir must stay inside the run directory.")
    wandb_dir = Path(run_dir) / relative_dir
    wandb_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "enabled": True,
        "active": False,
        "mode": mode,
        "project": identity["project"],
        "name": identity["name"],
        "group": identity["group"],
        "run_id": None,
        "url": None,
        "dir": str(wandb_dir.resolve()),
        "warning": None,
    }
    state = {
        "run": None,
        "module": None,
        "run_dir": Path(run_dir),
        "strict": bool(config.tracking.wandb.strict),
        "metadata": metadata,
    }
    try:
        wandb = importlib.import_module("wandb")
        entity = str(config.tracking.wandb.entity or "").strip() or None
        notes = str(config.tracking.wandb.notes or "").strip() or None
        run = wandb.init(
            project=identity["project"],
            entity=entity,
            dir=str(wandb_dir),
            mode=mode,
            name=identity["name"],
            group=identity["group"],
            job_type="image_train",
            notes=notes,
            tags=list(config.tracking.wandb.tags),
            config=config_to_dict(config),
            reinit="finish_previous",
        )
        state["run"] = run
        state["module"] = wandb
        if hasattr(wandb, "define_metric"):
            wandb.define_metric("train/step")
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("generation/step")
            wandb.define_metric("generation/*", step_metric="generation/step")

        metadata.update(
            {
                "active": True,
                "run_id": getattr(run, "id", None),
                "url": getattr(run, "url", None),
            }
        )
        update_run_record(run_dir, wandb=metadata)
        target = metadata["url"] or metadata["run_id"] or identity["name"]
        print(f"W&B active: {target}", flush=True)
        return state
    except Exception as exc:
        return _wandb_failure(state, exc, "initialization")


def log_wandb(run, values):
    """Log one scalar or image payload and return the active run state."""
    if run is None:
        return None
    try:
        payload = {
            key: run["module"].Image(str(value)) if isinstance(value, Path) else value
            for key, value in values.items()
        }
        run["run"].log(payload)
        return run
    except Exception as exc:
        return _wandb_failure(run, exc, "logging")


def finish_wandb(run, success):
    """Finish one active W&B run with the training outcome."""
    if run is None or run["run"] is None:
        return
    try:
        run["run"].finish(exit_code=0 if success else 1)
        metadata = {**run["metadata"], "active": False, "success": bool(success)}
        update_run_record(run["run_dir"], wandb=metadata)
    except Exception as exc:
        _wandb_failure(run, exc, "finish")

"""Exact training-state recovery for long runner jobs."""

import fcntl
import random
import signal
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from mem_param_src.runners.artifacts import (
    file_sha256,
    load_torch,
    read_json,
    save_torch,
    write_json,
)


RECOVERY_FORMAT_VERSION = 1


class StatefulBatchSampler:
    """Yield shuffled batches while exposing the exact permutation cursor."""

    def __init__(self, n_samples, batch_size, seed, state=None):
        self.n_samples = int(n_samples)
        self.batch_size = int(batch_size)
        if self.n_samples < 1:
            raise ValueError("StatefulBatchSampler requires at least one sample.")
        if self.batch_size < 1:
            raise ValueError("StatefulBatchSampler batch_size must be positive.")
        self.generator = torch.Generator().manual_seed(int(seed))
        self.order = torch.empty(0, dtype=torch.long)
        self.cursor = 0
        if state is not None:
            self.load_state_dict(state)

    def __iter__(self):
        while True:
            if self.cursor >= len(self.order):
                self.order = torch.randperm(
                    self.n_samples,
                    generator=self.generator,
                )
                self.cursor = 0
            end = min(self.cursor + self.batch_size, self.n_samples)
            indices = self.order[self.cursor : end]
            self.cursor = end
            yield indices.tolist()

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def state_dict(self):
        """Return the RNG, permutation, and next unread index."""
        return {
            "n_samples": self.n_samples,
            "batch_size": self.batch_size,
            "generator_state": self.generator.get_state(),
            "order": self.order.clone(),
            "cursor": int(self.cursor),
        }

    def load_state_dict(self, state):
        """Restore one previously saved batch position."""
        if int(state["n_samples"]) != self.n_samples:
            raise ValueError("Recovery sample count does not match the saved run.")
        if int(state["batch_size"]) != self.batch_size:
            raise ValueError("Recovery physical batch size does not match the saved run.")
        order = state["order"].to(dtype=torch.long, device="cpu")
        cursor = int(state["cursor"])
        if order.numel() not in (0, self.n_samples):
            raise ValueError("Recovery permutation has the wrong length.")
        if cursor < 0 or cursor > order.numel():
            raise ValueError("Recovery permutation cursor is invalid.")
        self.generator.set_state(state["generator_state"].cpu())
        self.order = order.clone()
        self.cursor = cursor


class StopRequest:
    """Convert termination signals into a stop at an optimizer boundary."""

    def __init__(self):
        self.requested = False
        self.signal_name = None
        self._previous = {}

    def _handle(self, signum, frame):
        del frame
        self.requested = True
        self.signal_name = signal.Signals(signum).name

    def install(self):
        """Install lightweight handlers for Slurm warning and termination signals."""
        for signum in (signal.SIGUSR1, signal.SIGTERM):
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle)
        return self

    def restore(self):
        """Restore handlers that were active before training."""
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)
        self._previous.clear()


def acquire_training_lock(run_dir):
    """Prevent two processes from training or resuming the same run."""
    path = Path(run_dir) / ".training.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(f"Another process holds the training lock: {path}") from exc
    return handle


def release_training_lock(handle):
    """Release a lock returned by acquire_training_lock."""
    if handle is None:
        return
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def capture_rng_state():
    """Capture every random stream used by the training process."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state):
    """Restore random streams after model and optimizer construction."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("Recovery contains CUDA RNG state but CUDA is unavailable.")
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("Recovery CUDA device count does not match this process.")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


@contextmanager
def preserve_rng_state():
    """Prevent monitoring and tracking callbacks from changing training RNGs."""
    state = capture_rng_state()
    try:
        yield
    finally:
        restore_rng_state(state)


def _recovery_dir(run_dir):
    path = Path(run_dir) / "recovery"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_recovery_state(
    run_dir,
    *,
    step,
    model,
    optimizer,
    batch_sampler,
    history,
    elapsed_seconds,
    wandb_run_id,
):
    """Atomically replace one of two rotating complete recovery states."""
    run_dir = Path(run_dir)
    recovery_dir = _recovery_dir(run_dir)
    index_path = recovery_dir / "index.json"
    index = read_json(index_path) if index_path.exists() else {"next_slot": 0, "slots": {}}
    slot = int(index.get("next_slot", 0)) % 2
    filename = f"state_{slot}.pt"
    path = recovery_dir / filename
    payload = {
        "recovery_format_version": RECOVERY_FORMAT_VERSION,
        "step": int(step),
        "config_sha256": file_sha256(run_dir / "config.json"),
        "data_bundle_sha256": file_sha256(run_dir / "data_bundle.pt"),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": capture_rng_state(),
        "batch_sampler": batch_sampler.state_dict(),
        "history": list(history),
        "elapsed_seconds": float(elapsed_seconds),
        "wandb_run_id": wandb_run_id,
    }
    save_torch(path, payload)
    index["slots"][str(slot)] = {
        "path": filename,
        "step": int(step),
        "sha256": file_sha256(path),
    }
    index["next_slot"] = 1 - slot
    write_json(index_path, index)
    return path


def load_latest_recovery(run_dir):
    """Load the newest valid rotating state and verify saved-run identity."""
    run_dir = Path(run_dir)
    recovery_dir = run_dir / "recovery"
    candidates = []
    for path in sorted(recovery_dir.glob("state_*.pt")):
        try:
            payload = load_torch(path)
            if int(payload.get("recovery_format_version", -1)) != RECOVERY_FORMAT_VERSION:
                continue
            if payload.get("config_sha256") != file_sha256(run_dir / "config.json"):
                continue
            if payload.get("data_bundle_sha256") != file_sha256(
                run_dir / "data_bundle.pt"
            ):
                continue
            candidates.append((int(payload["step"]), path, payload))
        except Exception:
            continue
    if not candidates:
        raise FileNotFoundError(f"No valid recovery state exists under: {recovery_dir}")
    return max(candidates, key=lambda item: item[0])


__all__ = [
    "StatefulBatchSampler",
    "StopRequest",
    "acquire_training_lock",
    "capture_rng_state",
    "load_latest_recovery",
    "preserve_rng_state",
    "release_training_lock",
    "restore_rng_state",
    "save_recovery_state",
]

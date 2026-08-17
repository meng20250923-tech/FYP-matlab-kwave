"""Dataset and metric helpers shared by the Test4 MNIST experiment scripts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST

from pat_fno.config.mnist_pat import (
    CFL,
    CONDITIONS,
    DX,
    DY,
    GRID_SIZE,
    MEDIUM_DENSITY,
    SEED,
    SOUND_SPEED,
    SPLITS,
)
from pat_fno.operators.fourier.numpy_reference import numpy_forward_2d
from pat_fno.operators.kwave import kwave_forward_2d

ROOT = Path(__file__).resolve().parents[3]


def build_setting(condition: str) -> SimpleNamespace:
    """Construct the numerical setting for an acquisition condition."""
    spec = CONDITIONS[condition]
    dt = CFL * DX / SOUND_SPEED
    nt = int(math.ceil(math.hypot(GRID_SIZE * DX, GRID_SIZE * DY) / (SOUND_SPEED * dt)))
    return SimpleNamespace(
        Nx=GRID_SIZE,
        Ny=GRID_SIZE,
        dx=DX,
        dy=DY,
        soundSpeed=SOUND_SPEED,
        mediumDensity=MEDIUM_DENSITY,
        CFL=CFL,
        dt=dt,
        Nt=nt,
        t_array=np.arange(nt, dtype=np.float64) * dt,
        kwaveBoundary=spec["boundary"],
        computation=SimpleNamespace(
            theta_max=np.deg2rad(spec["theta_deg"]),
            interpolationMethodF="cubic",
        ),
    )


def conditions(selected: str) -> list[str]:
    """Resolve one condition or all configured acquisition conditions."""
    return list(CONDITIONS) if selected == "all" else [selected]


def load_mnist_splits(
    raw_root: Path,
    limits: dict[str, int] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return deterministic MNIST images, labels, and source indices."""
    limits = limits or SPLITS
    train = MNIST(raw_root, train=True, download=True)
    test = MNIST(raw_root, train=False, download=True)
    rng = np.random.default_rng(SEED)
    train_perm = rng.permutation(len(train.data))
    test_perm = rng.permutation(len(test.data))
    start = 0
    output = {}

    for split in ("train", "validation"):
        size = limits[split]
        indices = train_perm[start : start + size]
        start += size
        output[split] = (
            train.data[indices].numpy(),
            train.targets[indices].numpy(),
            indices.astype(np.int64),
        )

    size = limits["test"]
    indices = test_perm[:size]
    output["test"] = (
        test.data[indices].numpy(),
        test.targets[indices].numpy(),
        indices.astype(np.int64),
    )
    return output


def to_pressure(image: np.ndarray) -> np.ndarray:
    """Resize and normalise an MNIST image as an initial pressure field."""
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0)[None, None]
    resized = F.interpolate(
        tensor,
        size=(GRID_SIZE, GRID_SIZE),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    maximum = torch.max(resized)
    normalised = resized / maximum if maximum > 0 else resized
    return normalised.numpy().astype(np.float32)


def generate_sample(
    p0: np.ndarray,
    setting: SimpleNamespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate Fourier and k-Wave pressure data for one image."""
    raw = numpy_forward_2d(p0, setting).astype(np.float32)
    scaled = (setting.soundSpeed * raw).astype(np.float32)
    kwave = kwave_forward_2d(p0, setting).astype(np.float32)
    if kwave.shape != scaled.shape:
        raise ValueError(f"k-Wave shape {kwave.shape} does not match Fourier shape {scaled.shape}")
    return raw, scaled, kwave


def write_shard(
    path: Path,
    p0: np.ndarray,
    raw: np.ndarray,
    scaled: np.ndarray,
    kwave: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    attributes: dict,
) -> None:
    """Write one shard using an atomic file replacement."""
    temporary = path.with_suffix(".partial")
    datasets = {
        "p0": p0,
        "fourier_raw": raw,
        "data_fft": scaled,
        "kwave_forward": kwave,
        "label": labels,
        "source_index": indices,
    }

    with h5py.File(temporary, "w") as handle:
        for name, values in datasets.items():
            handle.create_dataset(
                name,
                data=values,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
        for key, value in attributes.items():
            handle.attrs[key] = value
        handle.attrs["complete"] = True

    temporary.replace(path)


def shard_paths(
    dataset_root: Path,
    condition: str,
    split: str,
) -> list[Path]:
    """Return sorted HDF5 shard paths for a dataset split."""
    return sorted((dataset_root / condition).glob(f"{split}_*.h5"))


def load_arrays(
    dataset_root: Path,
    condition: str,
    split: str,
) -> dict[str, np.ndarray]:
    """Load and concatenate every completed shard in a dataset split."""
    names = ("p0", "fourier_raw", "data_fft", "kwave_forward", "label", "source_index")
    chunks = {name: [] for name in names}
    paths = shard_paths(dataset_root, condition, split)
    if not paths:
        raise FileNotFoundError(f"No {condition}/{split} shards in {dataset_root}")
    for path in paths:
        with h5py.File(path, "r") as handle:
            if not handle.attrs.get("complete", False):
                continue
            for name in names:
                chunks[name].append(handle[name][...])
    return {name: np.concatenate(values, axis=0) for name, values in chunks.items()}


def rel_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    """Compute relative L2 error for one prediction."""
    numerator = np.linalg.norm(prediction - target)
    denominator = max(
        np.linalg.norm(target),
        np.finfo(float).eps,
    )
    return float(numerator / denominator)


def centered_corr(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    """Compute centred correlation for one prediction."""
    prediction = prediction.ravel().astype(float)
    target = target.ravel().astype(float)
    prediction -= prediction.mean()
    target -= target.mean()

    numerator = np.dot(prediction, target)
    denominator = max(
        np.linalg.norm(prediction) * np.linalg.norm(target),
        np.finfo(float).eps,
    )
    return float(numerator / denominator)


def batch_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    """Compute aggregate prediction metrics for a batch."""
    pairs = zip(prediction, target, strict=False)
    relative_errors = [rel_l2(p, t) for p, t in pairs]

    pairs = zip(prediction, target, strict=False)
    correlations = [centered_corr(p, t) for p, t in pairs]

    return {
        "rel_l2_mean": float(np.mean(relative_errors)),
        "centered_corr_mean": float(np.mean(correlations)),
        "mse": float(np.mean((prediction - target) ** 2)),
    }


def save_json(path: Path, payload: dict) -> None:
    """Write a JSON payload after creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)

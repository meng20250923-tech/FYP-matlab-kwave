"""Train the three Test4 forward-operator baselines on a paired MNIST dataset.

The script deliberately trains one boundary condition at a time.  That makes every
reported number traceable to a single physical setup instead of hiding boundary
effects in a mixed training distribution.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from pat_fno.data.mnist import (
    CONDITIONS,
    ROOT,
    batch_metrics,
    build_setting,
    load_arrays,
    save_json,
    shard_paths,
)
from pat_fno.models import TinyFNO2d

SCENARIOS = ("fno_only", "fourier_to_fno", "fno_to_fourier")


def load_arrays_compact(dataset_root, condition, split):
    """Load a large split without its unused fourier_raw field."""
    names = ("p0", "data_fft", "kwave_forward", "label", "source_index")
    chunks = {name: [] for name in names}
    paths = shard_paths(dataset_root, condition, split)
    if not paths:
        raise FileNotFoundError(f"No {condition}/{split} shards in {dataset_root}")
    for path in paths:
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("complete", False):
                for name in names:
                    chunks[name].append(handle[name][...])
    return {name: np.concatenate(chunks[name], axis=0) for name in names}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--scale", choices=("medium", "large"), default="medium")
    parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument(
        "--batched-fourier",
        action="store_true",
        help="Use the batched differentiable Fourier operator; only for fno_to_fourier.",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help="Deterministically use the first N shuffled training samples.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Explicit result directory. Required for custom experiment layouts.",
    )
    return parser.parse_args()


def _device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable.")
    return torch.device(
        "cuda"
        if requested == "auto" and torch.cuda.is_available()
        else requested
        if requested != "auto"
        else "cpu"
    )


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _as_tensor(values: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(values.astype(np.float32, copy=False))


def _resize_p0(values: torch.Tensor, output_hw: tuple[int, int]) -> torch.Tensor:
    return torch.nn.functional.interpolate(
        values[:, None], size=output_hw, mode="bilinear", align_corners=False
    )[:, 0]


class ForwardDataset(torch.utils.data.Dataset):
    """Present normalized source, target, and pressure tensors for FNO training."""

    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        scenario: str,
        p0_mean: float,
        p0_std: float,
        data_mean: float,
        data_std: float,
    ):
        self.p0 = _as_tensor(arrays["p0"])
        self.fourier = _as_tensor(arrays["data_fft"])
        self.kwave = _as_tensor(arrays["kwave_forward"])
        self.scenario = scenario
        self.p0_mean, self.p0_std = p0_mean, p0_std
        self.data_mean, self.data_std = data_mean, data_std

    def __len__(self) -> int:
        return self.p0.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p0, fourier, kwave = self.p0[index], self.fourier[index], self.kwave[index]
        if self.scenario == "fno_only":
            source = _resize_p0(p0[None], tuple(kwave.shape))[0]
            source = (source - self.p0_mean) / self.p0_std
        elif self.scenario == "fourier_to_fno":
            source = (fourier - self.data_mean) / self.data_std
        else:
            source = (p0 - self.p0_mean) / self.p0_std
        target = (kwave - self.data_mean) / self.data_std
        return source, target, p0


def _forward_fourier(
    model: TinyFNO2d, p0: torch.Tensor, normalized_p0: torch.Tensor, setting
) -> torch.Tensor:
    """Learn a pressure correction and run the differentiable Fourier operator."""
    from pat_fno.operators.fourier import fpat_forward_2d

    # A residual parametrisation starts exactly from the physical image and makes
    # optimisation substantially more stable than asking a small FNO to recreate it.
    corrected = p0 + 0.10 * model(normalized_p0[:, None])[:, 0]
    output = [
        setting.soundSpeed
        * fpat_forward_2d(
            item,
            setting.computation.theta_max,
            setting.soundSpeed,
            setting.Nt,
            (setting.dx, setting.dy),
            setting.dt,
        )
        for item in corrected
    ]
    return torch.stack(output)


def _forward_fourier_batched(
    model: TinyFNO2d, p0: torch.Tensor, normalized_p0: torch.Tensor, setting
) -> torch.Tensor:
    from pat_fno.operators.fourier import fpat_forward_2d_batched

    corrected = p0 + 0.10 * model(normalized_p0[:, None])[:, 0]
    return setting.soundSpeed * fpat_forward_2d_batched(
        corrected,
        setting.computation.theta_max,
        setting.soundSpeed,
        setting.Nt,
        (setting.dx, setting.dy),
        setting.dt,
    )


_FORWARD_FOURIER = _forward_fourier


def _model_prediction(
    model: TinyFNO2d,
    scenario: str,
    source: torch.Tensor,
    p0: torch.Tensor,
    setting,
    data_mean: float,
    data_std: float,
) -> torch.Tensor:
    if scenario == "fno_to_fourier":
        physical = _FORWARD_FOURIER(model, p0, source, setting)
        return (physical - data_mean) / data_std
    return model(source[:, None])[:, 0]


@torch.no_grad()
def _evaluate(
    model: TinyFNO2d,
    loader: DataLoader,
    scenario: str,
    setting,
    data_mean: float,
    data_std: float,
    device: torch.device,
    return_arrays: bool = False,
):
    model.eval()
    predictions, targets = [], []
    squared_error = 0.0
    count = 0
    for source, target, p0 in loader:
        source, target, p0 = source.to(device), target.to(device), p0.to(device)
        prediction = _model_prediction(model, scenario, source, p0, setting, data_mean, data_std)
        squared_error += torch.sum((prediction - target) ** 2).item()
        count += target.numel()
        if return_arrays:
            predictions.append((prediction * data_std + data_mean).cpu().numpy())
            targets.append((target * data_std + data_mean).cpu().numpy())
    result = {"normalized_mse": squared_error / max(count, 1)}
    if return_arrays:
        prediction = np.concatenate(predictions)
        target = np.concatenate(targets)
        result.update(batch_metrics(prediction, target))
        result["prediction"] = prediction
        result["target"] = target
    return result


def _validate_args(args: argparse.Namespace) -> None:
    """Validate combinations that argparse cannot express directly."""
    if args.batched_fourier and args.scenario != "fno_to_fourier":
        raise ValueError("--batched-fourier is only valid with --scenario fno_to_fourier.")


def _configure_fourier_forward(use_batched: bool) -> None:
    """Select the established scalar or batched differentiable Fourier path."""
    global _FORWARD_FOURIER
    _FORWARD_FOURIER = _forward_fourier_batched if use_batched else _forward_fourier


def _select_training_subset(
    arrays: dict[str, np.ndarray],
    sample_count: int | None,
    seed: int,
) -> dict[str, np.ndarray]:
    """Select the deterministic shuffled-prefix training subset."""
    if sample_count is None:
        return arrays
    available = len(arrays["p0"])
    if not 0 < sample_count <= available:
        raise ValueError(f"--train-samples must be in [1, {available}].")
    subset_rng = np.random.default_rng(seed)
    subset_indices = np.sort(subset_rng.permutation(available)[:sample_count])
    return {name: values[subset_indices] for name, values in arrays.items()}


def _normalization(train: dict[str, np.ndarray]) -> dict[str, float]:
    """Calculate image- and data-domain statistics from the training split."""
    return {
        "p0_mean": float(train["p0"].mean()),
        "p0_std": float(train["p0"].std() + 1e-6),
        "data_mean": float(train["kwave_forward"].mean()),
        "data_std": float(train["kwave_forward"].std() + 1e-6),
    }


def _build_loaders(
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    scenario: str,
    normalization: dict[str, float],
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build loaders while preserving training shuffle and evaluation order."""
    statistics = (
        normalization["p0_mean"],
        normalization["p0_std"],
        normalization["data_mean"],
        normalization["data_std"],
    )
    train_set = ForwardDataset(train, scenario, *statistics)
    validation_set = ForwardDataset(validation, scenario, *statistics)
    test_set = ForwardDataset(test, scenario, *statistics)
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            generator=generator,
        ),
        DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0),
    )


def _result_root(args: argparse.Namespace, dataset_name: str) -> Path:
    """Resolve the stable output directory for a requested training run."""
    if args.output_root is not None:
        return args.output_root
    if args.train_samples is not None:
        return (
            ROOT
            / "results"
            / "sample_efficiency"
            / dataset_name
            / args.condition
            / args.scenario
            / f"n{args.train_samples}_seed{args.seed}"
        )
    return ROOT / "results" / "mnist_medium" / dataset_name / args.condition / args.scenario


def _train_epoch(
    model: TinyFNO2d,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    scenario: str,
    setting,
    data_mean: float,
    data_std: float,
    device: torch.device,
) -> float:
    """Run one optimisation epoch and return its elementwise normalised MSE."""
    model.train()
    loss_sum = 0.0
    elements = 0
    for source, target, p0 in loader:
        source, target, p0 = source.to(device), target.to(device), p0.to(device)
        optimiser.zero_grad(set_to_none=True)
        prediction = _model_prediction(
            model,
            scenario,
            source,
            p0,
            setting,
            data_mean,
            data_std,
        )
        loss = torch.mean((prediction - target) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        loss_sum += torch.sum((prediction - target) ** 2).item()
        elements += target.numel()
    return loss_sum / max(elements, 1)


def _save_checkpoint(
    path: Path,
    model: TinyFNO2d,
    arguments: dict[str, object],
    normalization: dict[str, float],
    setting,
) -> None:
    """Save a checkpoint with the established field names and payload types."""
    torch.save(
        {
            "model": model.state_dict(),
            "arguments": arguments,
            "normalization": normalization,
            "setting": vars(setting),
        },
        path,
    )


def _train_model(
    model: TinyFNO2d,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    args: argparse.Namespace,
    setting,
    normalization: dict[str, float],
    device: torch.device,
    result_root: Path,
    arguments_record: dict[str, object],
) -> tuple[list[dict[str, float]], float]:
    """Train, validate, and retain the best model checkpoint."""
    history: list[dict[str, float]] = []
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_mse = _train_epoch(
            model,
            train_loader,
            optimiser,
            args.scenario,
            setting,
            normalization["data_mean"],
            normalization["data_std"],
            device,
        )
        scheduler.step()
        validation_metrics = _evaluate(
            model,
            validation_loader,
            args.scenario,
            setting,
            normalization["data_mean"],
            normalization["data_std"],
            device,
        )
        record = {
            "epoch": epoch,
            "train_normalized_mse": train_mse,
            **validation_metrics,
            "lr": optimiser.param_groups[0]["lr"],
        }
        history.append(record)
        print(
            f"epoch {epoch:03d}/{args.epochs}: "
            f"train={record['train_normalized_mse']:.6e}, "
            f"validation={record['normalized_mse']:.6e}"
        )
        if validation_metrics["normalized_mse"] < best:
            best = validation_metrics["normalized_mse"]
            _save_checkpoint(
                result_root / "best.pt",
                model,
                arguments_record,
                normalization,
                setting,
            )
    return history, best


def _save_test_outputs(
    result_root: Path,
    test: dict[str, np.ndarray],
    prediction: np.ndarray,
    target: np.ndarray,
) -> None:
    """Save physical test predictions and their sample identifiers."""
    np.savez_compressed(
        result_root / "test_predictions.npz",
        prediction=prediction,
        target=target,
        p0=test["p0"],
        label=test["label"],
        source_index=test["source_index"],
    )


def main() -> None:
    """Train and evaluate one configured forward-operator surrogate."""
    args = _parse_args()
    arguments_record = {
        name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()
    }
    _validate_args(args)
    _configure_fourier_forward(args.batched_fourier)
    dataset_name = args.dataset or f"mnist_{args.scale}_v1"
    _seed(args.seed)
    device = _device(args.device)
    dataset_root = ROOT / "datasets" / dataset_name
    array_loader = load_arrays_compact if args.scale == "large" else load_arrays
    train = array_loader(dataset_root, args.condition, "train")
    train = _select_training_subset(train, args.train_samples, args.seed)
    validation = array_loader(dataset_root, args.condition, "validation")
    test = array_loader(dataset_root, args.condition, "test")
    normalization = _normalization(train)
    train_loader, validation_loader, test_loader = _build_loaders(
        train,
        validation,
        test,
        args.scenario,
        normalization,
        args.batch_size,
        args.seed,
    )
    model = TinyFNO2d(args.modes, args.modes, args.width, args.layers).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    setting = build_setting(args.condition)
    result_root = _result_root(args, dataset_name)
    result_root.mkdir(parents=True, exist_ok=True)
    history, best = _train_model(
        model,
        train_loader,
        validation_loader,
        optimiser,
        scheduler,
        args,
        setting,
        normalization,
        device,
        result_root,
        arguments_record,
    )
    checkpoint = torch.load(result_root / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    metrics = _evaluate(
        model,
        test_loader,
        args.scenario,
        setting,
        normalization["data_mean"],
        normalization["data_std"],
        device,
        return_arrays=True,
    )
    prediction, target = metrics.pop("prediction"), metrics.pop("target")
    _save_test_outputs(result_root, test, prediction, target)
    save_json(result_root / "history.json", {"arguments": arguments_record, "history": history})
    save_json(
        result_root / "metrics.json",
        {
            "dataset": dataset_name,
            "condition": args.condition,
            "scenario": args.scenario,
            "test": metrics,
            "normalization": checkpoint["normalization"],
            "best_validation_normalized_mse": best,
        },
    )
    print(f"Saved model and test metrics: {result_root}")
    print(metrics)


if __name__ == "__main__":
    main()

"""Benchmark end-to-end PAT forward operators without modifying experiment data.

The benchmark writes only to results/evaluation/<dataset>/runtime. Learned
pipelines include their required Fourier preprocessing/postprocessing so their
reported times are comparable at the pipeline level.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pat_fno.data.mnist import CONDITIONS, ROOT, build_setting, conditions, load_arrays
from pat_fno.models import TinyFNO2d
from pat_fno.operators.fourier import fpat_forward_2d_batched
from pat_fno.operators.kwave import kwave_forward_2d
from scripts.train.train_forward_operator import _resize_p0

SCENARIOS = ("fno_only", "fourier_to_fno", "fno_to_fourier")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the runtime benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mnist_medium_v1")
    parser.add_argument("--condition", choices=("all", *CONDITIONS), default="all")
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--include-kwave", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    """Wait for queued CUDA work before recording elapsed time."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_callable(function, warmup: int, repeats: int, device: torch.device) -> list[float]:
    """Return repeated wall-clock durations after unrecorded warm-up calls."""
    for _ in range(warmup):
        function()
    synchronize(device)
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        synchronize(device)
        durations.append(time.perf_counter() - start)
    return durations


def summary(method: str, durations: list[float], count: int, **extra) -> dict:
    """Summarise per-call durations as per-sample runtime statistics."""
    per_sample = [value / count for value in durations]
    mean = statistics.mean(per_sample)
    std = statistics.stdev(per_sample) if len(per_sample) > 1 else 0.0
    return {
        "method": method,
        "samples_per_call": count,
        "mean_ms_per_sample": 1e3 * mean,
        "std_ms_per_sample": 1e3 * std,
        "throughput_samples_per_s": 1.0 / mean,
        **extra,
    }


def load_model(path: Path, device: torch.device) -> tuple[TinyFNO2d, dict]:
    """Load one trained FNO and its complete checkpoint metadata."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    arguments = checkpoint["arguments"]
    model = TinyFNO2d(
        arguments["modes"], arguments["modes"], arguments["width"], arguments["layers"]
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def fourier(p0: torch.Tensor, setting) -> torch.Tensor:
    """Evaluate the scaled analytical Fourier forward operator."""
    return setting.soundSpeed * fpat_forward_2d_batched(
        p0,
        setting.computation.theta_max,
        setting.soundSpeed,
        setting.Nt,
        (setting.dx, setting.dy),
        setting.dt,
    )


def validate_arguments(args: argparse.Namespace) -> None:
    """Reject invalid benchmark sizes before any data are loaded."""
    if args.batch_size <= 0 or args.max_samples <= 0 or args.repeats <= 0 or args.warmup < 0:
        raise ValueError(
            "Batch size, samples and repeats must be positive; warmup must be non-negative."
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicit or automatic PyTorch device request."""
    return torch.device(
        "cuda"
        if requested == "auto" and torch.cuda.is_available()
        else "cpu"
        if requested == "auto"
        else requested
    )


def benchmark_fourier(
    p0: torch.Tensor,
    setting: Any,
    args: argparse.Namespace,
    device: torch.device,
    count: int,
) -> dict:
    """Benchmark the analytical Fourier operator."""
    with torch.inference_mode():
        durations = time_callable(
            lambda p0=p0, setting=setting: fourier(p0, setting),
            args.warmup,
            args.repeats,
            device,
        )
    return summary(
        "Fourier", durations, count, device=device.type, parameters=0, checkpoint_bytes=0
    )


def learned_pipeline(
    scenario: str,
    p0: torch.Tensor,
    setting: Any,
    output_hw: tuple[int, int],
    norm: dict,
    model: TinyFNO2d,
) -> Callable[[], torch.Tensor]:
    """Construct the timed end-to-end pipeline for one learned scenario."""

    def pipeline(
        scenario=scenario,
        p0=p0,
        setting=setting,
        output_hw=output_hw,
        norm=norm,
        model=model,
    ):
        with torch.inference_mode():
            if scenario == "fno_only":
                source = (_resize_p0(p0, output_hw) - norm["p0_mean"]) / norm["p0_std"]
                return model(source[:, None])[:, 0]
            if scenario == "fourier_to_fno":
                source = (fourier(p0, setting) - norm["data_mean"]) / norm["data_std"]
                return model(source[:, None])[:, 0]
            source = (p0 - norm["p0_mean"]) / norm["p0_std"]
            corrected = p0 + 0.10 * model(source[:, None])[:, 0]
            return fourier(corrected, setting)

    return pipeline


def benchmark_learned_operators(
    checkpoint_root: Path,
    condition: str,
    arrays: dict[str, np.ndarray],
    p0: torch.Tensor,
    setting: Any,
    args: argparse.Namespace,
    device: torch.device,
    count: int,
) -> list[dict]:
    """Benchmark each available learned or hybrid forward operator."""
    rows = []
    for scenario in SCENARIOS:
        checkpoint_path = checkpoint_root / condition / scenario / "best.pt"
        if not checkpoint_path.exists():
            print(f"Missing checkpoint, skip {scenario}: {checkpoint_path}")
            continue
        model, checkpoint = load_model(checkpoint_path, device)
        norm = checkpoint["normalization"]
        output_hw = tuple(arrays["kwave_forward"].shape[-2:])
        pipeline = learned_pipeline(scenario, p0, setting, output_hw, norm, model)
        durations = time_callable(pipeline, args.warmup, args.repeats, device)
        rows.append(
            summary(
                scenario,
                durations,
                count,
                device=device.type,
                parameters=sum(parameter.numel() for parameter in model.parameters()),
                checkpoint_bytes=checkpoint_path.stat().st_size,
            )
        )
    return rows


def benchmark_kwave(
    p0_np: np.ndarray,
    setting: Any,
    args: argparse.Namespace,
    count: int,
) -> dict:
    """Benchmark the CPU NumPy k-Wave implementation."""

    def kwave_pipeline(setting=setting, p0_np=p0_np):
        return [kwave_forward_2d(item, setting) for item in p0_np]

    durations = time_callable(kwave_pipeline, args.warmup, args.repeats, torch.device("cpu"))
    return summary("k-Wave", durations, count, device="cpu", parameters=0, checkpoint_bytes=0)


def add_comparisons(rows: list[dict], condition: str) -> None:
    """Append k-Wave speed-up and acquisition-condition fields in place."""
    kwave_row = next((row for row in rows if row["method"] == "k-Wave"), None)
    for row in rows:
        row["speedup_vs_kwave"] = (
            kwave_row["mean_ms_per_sample"] / row["mean_ms_per_sample"] if kwave_row else None
        )
        row["condition"] = condition


def benchmark_metadata(
    args: argparse.Namespace,
    condition: str,
    device: torch.device,
    count: int,
    rows: list[dict],
) -> dict:
    """Build the runtime metadata record in its stable output order."""
    return {
        "dataset": args.dataset,
        "condition": condition,
        "split": args.split,
        "device_requested": args.device,
        "device_used": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cpu": platform.processor(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "batch_size": count,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "rows": rows,
    }


def write_benchmark(
    output_root: Path,
    output_json: Path,
    condition: str,
    device: torch.device,
    metadata: dict,
    rows: list[dict],
) -> None:
    """Write one condition's JSON metadata and tabular CSV rows."""
    output_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with (output_root / f"{condition}_{device.type}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved runtime benchmark: {output_json}")


def benchmark_condition(
    args: argparse.Namespace,
    condition: str,
    device: torch.device,
    checkpoint_root: Path,
    output_root: Path,
) -> None:
    """Run and save every requested operator benchmark for one condition."""
    output_json = output_root / f"{condition}_{device.type}.json"
    if output_json.exists() and not args.overwrite:
        print(f"Skip existing benchmark: {output_json}")
        return
    arrays = load_arrays(ROOT / "datasets" / args.dataset, condition, args.split)
    count = min(args.batch_size, args.max_samples, len(arrays["p0"]))
    p0_np = arrays["p0"][:count].astype(np.float32, copy=False)
    p0 = torch.from_numpy(p0_np).to(device)
    setting = build_setting(condition)
    rows = [benchmark_fourier(p0, setting, args, device, count)]
    rows.extend(
        benchmark_learned_operators(
            checkpoint_root,
            condition,
            arrays,
            p0,
            setting,
            args,
            device,
            count,
        )
    )
    if args.include_kwave:
        rows.append(benchmark_kwave(p0_np, setting, args, count))
    add_comparisons(rows, condition)
    metadata = benchmark_metadata(args, condition, device, count, rows)
    write_benchmark(output_root, output_json, condition, device, metadata, rows)


def main() -> None:
    """Benchmark the requested operators and acquisition conditions."""
    args = parse_args()
    validate_arguments(args)
    device = resolve_device(args.device)
    checkpoint_root = args.checkpoint_root or ROOT / "results" / "mnist_medium" / args.dataset
    output_root = ROOT / "results" / "evaluation" / args.dataset / "runtime"
    output_root.mkdir(parents=True, exist_ok=True)
    for condition in conditions(args.condition):
        benchmark_condition(args, condition, device, checkpoint_root, output_root)


if __name__ == "__main__":
    main()

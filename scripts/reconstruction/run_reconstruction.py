"""Run any PAT reconstruction method from one consistent command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pat_fno.data.mnist import CONDITIONS, ROOT, batch_metrics, build_setting, conditions, rel_l2
from pat_fno.operators.fourier import fpat_forward_2d, numpy_inverse_2d
from pat_fno.operators.kwave import kwave_adjoint_2d, kwave_forward_2d, kwave_inverse_2d
from scripts.reconstruction.common import load_subsampled, reconstruction_tag, subsampled_path

METHODS = (
    "fourier",
    "time_reversal",
    "iterated_time_reversal",
    "gradient_descent",
    "learned",
    "adjoint",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--dataset", default="mnist_medium_v1")
    parser.add_argument("--condition", choices=("all", *CONDITIONS), default="all")
    parser.add_argument("--split", default="test")
    parser.add_argument("--keep-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--power-iterations", type=int, default=8)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--num-examples", type=int, default=5)
    parser.add_argument("--checkpoint-scale", choices=("medium", "large"), default="medium")
    parser.add_argument("--scenario", choices=("fno_only", "fourier_to_fno", "fno_to_fourier"))
    parser.add_argument("--learning-rate", type=float, default=3e-2)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _paths(args: argparse.Namespace, condition: str, directory: Path, suffix: str = ""):
    tag = reconstruction_tag(condition, args.split, args.keep_fraction, args.seed) + suffix
    return tag, directory / f"{tag}.npz", directory / f"{tag}_metrics.json"


def _skip(path: Path, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        print(f"Skip existing result: {path}")
        return True
    return False


def _save_examples(
    path: Path,
    target: np.ndarray,
    observed: np.ndarray,
    reconstructed: np.ndarray,
    errors: np.ndarray,
    label: str,
    keep_fraction: float,
    count: int,
) -> None:
    examples = min(count, len(target))
    figure, axes = plt.subplots(3, examples, figsize=(3.4 * examples, 8), squeeze=False)
    for index in range(examples):
        axes[0, index].imshow(target[index], cmap="viridis")
        axes[0, index].set_title(f"True p0, sample {index}")
        axes[1, index].imshow(observed[index], aspect="auto", cmap="viridis")
        axes[1, index].set_title(f"{keep_fraction:.0%} observed data")
        axes[2, index].imshow(reconstructed[index], cmap="viridis")
        axes[2, index].set_title(f"{label}, rel-L2={errors[index]:.3f}")
        for row in range(3):
            axes[row, index].axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _save_fourier_examples(
    path: Path,
    target: np.ndarray,
    reconstructed: np.ndarray,
    count: int,
) -> None:
    columns = min(count, len(target))
    figure, axes = plt.subplots(2, columns, figsize=(2.4 * columns, 4.5), squeeze=False)
    for index in range(columns):
        axes[0, index].imshow(target[index], cmap="gray")
        axes[0, index].set_title("truth")
        axes[1, index].imshow(reconstructed[index], cmap="gray")
        axes[1, index].set_title("Fourier inverse")
        for axis in axes[:, index]:
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _save_convergence(path: Path, residual: np.ndarray, error: np.ndarray) -> None:
    x = np.arange(residual.shape[1])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for axis, scale, title in (
        (axes[0], "semilogy", "Semilog convergence"),
        (axes[1], "loglog", "Loglog convergence"),
    ):
        start = 0 if scale == "semilogy" else 1
        getattr(axis, scale)(
            x[start:], residual.mean(axis=0)[start:], marker="o", label="data residual"
        )
        getattr(axis, scale)(
            x[start:], error.mean(axis=0)[start:], marker="s", label="reconstruction rel-L2"
        )
        axis.set(title=title, xlabel="iteration", ylabel="error / residual")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_fourier(args: argparse.Namespace, condition: str) -> None:
    output = ROOT / "results" / "reconstruction" / args.dataset / "fourier"
    output.mkdir(parents=True, exist_ok=True)
    tag, result_path, metrics_path = _paths(args, condition, output)
    if _skip(result_path, args.overwrite):
        return
    arrays = load_subsampled(subsampled_path(args, condition), args.max_samples, include_mask=True)
    setting = build_setting(condition)
    setting.computation.interpolationMethodI = "cubic"
    reconstruction = np.empty_like(arrays["p0"], dtype=np.float32)
    for index, measurement in enumerate(arrays["observed_data"]):
        reconstruction[index] = numpy_inverse_2d(measurement / setting.soundSpeed, setting).astype(
            np.float32
        )
        print(f"{condition}: {index + 1}/{len(reconstruction)}", flush=True)
    np.savez_compressed(
        result_path,
        reconstruction=reconstruction,
        p0=arrays["p0"],
        observed_data=arrays["observed_data"],
        mask=arrays["mask"],
        label=arrays["label"],
        source_index=arrays["source_index"],
    )
    metrics = {
        "method": "Fourier inverse",
        "condition": condition,
        "split": args.split,
        "keep_fraction": args.keep_fraction,
        "seed": args.seed,
        "num_samples": len(reconstruction),
        **batch_metrics(reconstruction, arrays["p0"]),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _save_fourier_examples(
        output / f"{tag}_examples.png",
        arrays["p0"],
        reconstruction,
        min(args.num_examples, len(reconstruction)),
    )


def run_direct(args: argparse.Namespace, condition: str, *, adjoint: bool) -> None:
    directory = "adjoint" if adjoint else "time_reversal"
    label = "Adjoint" if adjoint else "TR"
    method_name = "k-wave adjoint" if adjoint else "k-wave time reversal"
    operator = kwave_adjoint_2d if adjoint else kwave_inverse_2d
    output = ROOT / "results" / "reconstruction" / args.dataset / directory
    output.mkdir(parents=True, exist_ok=True)
    tag, result_path, metrics_path = _paths(args, condition, output)
    if _skip(result_path, args.overwrite):
        return
    arrays = load_subsampled(subsampled_path(args, condition), args.max_samples)
    setting = build_setting(condition)
    reconstruction = np.empty_like(arrays["p0"], dtype=np.float32)
    errors = np.empty(len(reconstruction), dtype=np.float64)
    for index, measurement in enumerate(arrays["observed_data"]):
        value = operator(measurement, setting)
        reconstruction[index] = value.astype(np.float32)
        errors[index] = rel_l2(value, arrays["p0"][index])
        print(
            f"{condition}: {index + 1}/{len(reconstruction)}, relative L2 = {errors[index]:.4f}",
            flush=True,
        )
    np.savez_compressed(
        result_path,
        reconstruction=reconstruction,
        p0=arrays["p0"],
        observed_data=arrays["observed_data"],
        label=arrays["label"],
        source_index=arrays["source_index"],
        relative_l2=errors,
    )
    metrics_path.write_text(
        json.dumps(
            {
                "method": method_name,
                "condition": condition,
                "split": args.split,
                "keep_fraction": args.keep_fraction,
                "seed": args.seed,
                "num_samples": len(reconstruction),
                "relative_l2_mean": float(errors.mean()),
                "relative_l2_std": float(errors.std()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _save_examples(
        output / f"{tag}_examples.png",
        arrays["p0"],
        arrays["observed_data"],
        reconstruction,
        errors,
        label,
        args.keep_fraction,
        args.num_examples,
    )


def estimate_lipschitz(mask: np.ndarray, setting: object, iterations: int) -> float:
    rng = np.random.default_rng(20260802)
    vector = rng.standard_normal((setting.Nx, setting.Ny))
    vector /= np.linalg.norm(vector)
    for _ in range(iterations):
        updated = kwave_adjoint_2d(mask * kwave_forward_2d(vector, setting), setting)
        norm = np.linalg.norm(updated)
        if norm <= np.finfo(float).eps:
            raise RuntimeError("Power iteration reached a zero vector.")
        vector = updated / norm
    updated = kwave_adjoint_2d(mask * kwave_forward_2d(vector, setting), setting)
    value = float(np.vdot(vector, updated).real)
    if value <= np.finfo(float).eps:
        raise RuntimeError(f"Invalid Lipschitz estimate: {value}")
    return value


def _iterative_method_configuration(
    args: argparse.Namespace,
    gradient_descent: bool,
) -> tuple[str, str]:
    """Return the output directory and filename suffix for an iterative method."""
    if not gradient_descent and args.step_size <= 0:
        raise ValueError("--step-size must be positive.")
    directory = "gradient_descent" if gradient_descent else "iterated_time_reversal"
    suffix = "" if gradient_descent else f"_step{args.step_size:g}"
    return directory, suffix


def _run_iterative_sample(
    *,
    target: np.ndarray,
    observed: np.ndarray,
    mask: np.ndarray,
    setting: object,
    iterations: int,
    gradient_descent: bool,
    step_size: float,
    power_iterations: int,
    progress_prefix: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None, float | None]:
    """Reconstruct one image and return its complete iteration history."""
    image = np.zeros_like(target, dtype=np.float64)
    lipschitz = None
    effective_step = None
    if gradient_descent:
        lipschitz = estimate_lipschitz(mask, setting, power_iterations)
        effective_step = 1.0 / lipschitz

    residual_history = np.empty(iterations + 1, dtype=np.float64)
    error_history = np.empty_like(residual_history)
    observed_norm = max(np.linalg.norm(mask * observed), np.finfo(float).eps)
    for iteration in range(iterations + 1):
        residual = mask * (kwave_forward_2d(image, setting) - observed)
        residual_history[iteration] = np.linalg.norm(residual) / observed_norm
        error_history[iteration] = rel_l2(image, target)
        print(
            f"{progress_prefix}, iteration {iteration:02d}/{iterations}, "
            f"residual={residual_history[iteration]:.4e}, "
            f"rel-L2={error_history[iteration]:.4f}",
            flush=True,
        )
        if iteration == iterations:
            break
        if gradient_descent:
            image -= effective_step * kwave_adjoint_2d(residual, setting)
        else:
            image -= step_size * kwave_inverse_2d(residual, setting)

    return image, residual_history, error_history, lipschitz, effective_step


def run_iterative(args: argparse.Namespace, condition: str, *, gradient_descent: bool) -> None:
    directory, suffix = _iterative_method_configuration(args, gradient_descent)
    output = ROOT / "results" / "reconstruction" / args.dataset / directory
    output.mkdir(parents=True, exist_ok=True)
    tag, result_path, metrics_path = _paths(args, condition, output, suffix)
    if _skip(result_path, args.overwrite):
        return
    arrays = load_subsampled(subsampled_path(args, condition), args.max_samples, include_mask=True)
    observed = arrays["observed_data"]
    masks = arrays["mask"].astype(np.float64)
    target = arrays["p0"]
    setting = build_setting(condition)
    count = len(target)
    reconstruction_all = np.empty_like(target, dtype=np.float32)
    residual_history = np.empty((count, args.iterations + 1), dtype=np.float64)
    error_history = np.empty_like(residual_history)
    lipschitz = np.empty(count, dtype=np.float64) if gradient_descent else None
    step_sizes = np.empty(count, dtype=np.float64) if gradient_descent else None
    for sample in range(count):
        if gradient_descent:
            print(f"{condition}: estimating L for sample {sample + 1}/{count}", flush=True)
        prefix = f"{condition}: sample {sample + 1}/{count}"
        image, residuals, errors, estimated_lipschitz, effective_step = _run_iterative_sample(
            target=target[sample],
            observed=observed[sample],
            mask=masks[sample],
            setting=setting,
            iterations=args.iterations,
            gradient_descent=gradient_descent,
            step_size=args.step_size,
            power_iterations=args.power_iterations,
            progress_prefix=prefix,
        )
        reconstruction_all[sample] = image.astype(np.float32)
        residual_history[sample] = residuals
        error_history[sample] = errors
        if gradient_descent:
            lipschitz[sample] = estimated_lipschitz
            step_sizes[sample] = effective_step
    payload = {
        "reconstruction": reconstruction_all,
        "p0": target,
        "observed_data": observed,
        "mask": masks,
        "label": arrays["label"],
        "source_index": arrays["source_index"],
        "residual_history": residual_history,
        "error_history": error_history,
    }
    if gradient_descent:
        payload.update(lipschitz=lipschitz, step_size=step_sizes)
    np.savez_compressed(result_path, **payload)
    metrics = {
        "method": "k-wave gradient descent"
        if gradient_descent
        else "iterated k-wave time reversal",
        "condition": condition,
        "split": args.split,
        "keep_fraction": args.keep_fraction,
        "seed": args.seed,
        "num_samples": count,
        "iterations": args.iterations,
        "final_relative_l2_mean": float(error_history[:, -1].mean()),
        "final_residual_mean": float(residual_history[:, -1].mean()),
    }
    if gradient_descent:
        metrics.update(
            power_iterations=args.power_iterations,
            lipschitz_mean=float(lipschitz.mean()),
            step_size_mean=float(step_sizes.mean()),
        )
    else:
        metrics["step_size"] = args.step_size
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _save_convergence(output / f"{tag}_convergence.png", residual_history, error_history)


def _load_learned_model(path: Path, device):
    import torch

    from pat_fno.models import TinyFNO2d

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    saved = checkpoint["arguments"]
    model = TinyFNO2d(saved["modes"], saved["modes"], saved["width"], saved["layers"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint["normalization"]


def _learned_prediction(model, scenario, image, setting, normalization):
    import torch.nn.functional as F

    p0_mean, p0_std = normalization["p0_mean"], normalization["p0_std"]
    data_mean, data_std = normalization["data_mean"], normalization["data_std"]
    if scenario == "fno_only":
        source = F.interpolate(
            image[None, None], size=(setting.Ny, setting.Nt), mode="bilinear", align_corners=False
        )
        return model((source - p0_mean) / p0_std)[0, 0] * data_std + data_mean
    if scenario == "fourier_to_fno":
        fourier = setting.soundSpeed * fpat_forward_2d(
            image,
            setting.computation.theta_max,
            setting.soundSpeed,
            setting.Nt,
            (setting.dx, setting.dy),
            setting.dt,
        )
        return model(((fourier - data_mean) / data_std)[None, None])[0, 0] * data_std + data_mean
    corrected = image + 0.10 * model(((image - p0_mean) / p0_std)[None, None])[0, 0]
    return setting.soundSpeed * fpat_forward_2d(
        corrected,
        setting.computation.theta_max,
        setting.soundSpeed,
        setting.Nt,
        (setting.dx, setting.dy),
        setting.dt,
    )


def _optimise_learned_sample(
    *,
    model,
    scenario: str,
    observation,
    mask,
    truth,
    setting: object,
    normalization: dict[str, float],
    iterations: int,
    learning_rate: float,
):
    """Optimise one image with a frozen learned forward operator."""
    import torch

    image = torch.zeros_like(truth, requires_grad=True)
    optimiser = torch.optim.Adam([image], lr=learning_rate)
    history = np.empty(iterations + 1, dtype=np.float64)
    for iteration in range(iterations + 1):
        estimate = _learned_prediction(model, scenario, image, setting, normalization)
        history[iteration] = float(
            torch.linalg.vector_norm(image.detach() - truth)
            / torch.clamp(torch.linalg.vector_norm(truth), min=torch.finfo(truth.dtype).eps)
        )
        if iteration == iterations:
            break
        loss = torch.mean((mask * (estimate - observation)) ** 2)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        with torch.no_grad():
            image.clamp_(0.0, 1.0)
    return image.detach(), history


def run_learned(args: argparse.Namespace, condition: str) -> None:
    import torch

    if args.scenario is None:
        raise ValueError("--scenario is required for --method learned.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device(args.device)
    output = (
        ROOT
        / "results"
        / "reconstruction"
        / args.dataset
        / "learned_operator"
        / args.checkpoint_scale
        / args.scenario
    )
    output.mkdir(parents=True, exist_ok=True)
    tag, result_path, metrics_path = _paths(args, condition, output)
    if _skip(result_path, args.overwrite):
        return
    arrays = load_subsampled(subsampled_path(args, condition), args.max_samples, include_mask=True)
    checkpoint_path = (
        ROOT
        / "results"
        / "mnist_medium"
        / f"mnist_{args.checkpoint_scale}_v1"
        / condition
        / args.scenario
        / "best.pt"
    )
    model, normalization = _load_learned_model(checkpoint_path, device)
    setting = build_setting(condition)
    count = len(arrays["p0"])
    reconstruction = np.empty_like(arrays["p0"], dtype=np.float32)
    histories = np.empty((count, args.iterations + 1), dtype=np.float64)
    for index in range(count):
        observation = torch.as_tensor(
            arrays["observed_data"][index], dtype=torch.float32, device=device
        )
        mask = torch.as_tensor(arrays["mask"][index], dtype=torch.float32, device=device)
        truth = torch.as_tensor(arrays["p0"][index], dtype=torch.float32, device=device)
        image, history = _optimise_learned_sample(
            model=model,
            scenario=args.scenario,
            observation=observation,
            mask=mask,
            truth=truth,
            setting=setting,
            normalization=normalization,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
        )
        histories[index] = history
        reconstruction[index] = image.cpu().numpy()
        print(
            f"{condition}/{args.scenario}: {index + 1}/{count}, rel-L2={histories[index, -1]:.4f}",
            flush=True,
        )
    np.savez_compressed(
        result_path,
        reconstruction=reconstruction,
        p0=arrays["p0"],
        mask=arrays["mask"],
        observed_data=arrays["observed_data"],
        label=arrays["label"],
        source_index=arrays["source_index"],
        error_history=histories,
    )
    metrics_path.write_text(
        json.dumps(
            {
                "condition": condition,
                "scenario": args.scenario,
                "checkpoint_scale": args.checkpoint_scale,
                "split": args.split,
                "num_samples": count,
                "iterations": args.iterations,
                "learning_rate": args.learning_rate,
                "final_relative_l2_mean": float(histories[:, -1].mean()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    for condition in conditions(args.condition):
        if args.method == "fourier":
            run_fourier(args, condition)
        elif args.method == "time_reversal":
            run_direct(args, condition, adjoint=False)
        elif args.method == "adjoint":
            run_direct(args, condition, adjoint=True)
        elif args.method == "gradient_descent":
            run_iterative(args, condition, gradient_descent=True)
        elif args.method == "iterated_time_reversal":
            run_iterative(args, condition, gradient_descent=False)
        else:
            run_learned(args, condition)


if __name__ == "__main__":
    main()

# ruff: noqa: E501
"""Generate the numerical LaTeX tables used in paper/thesis.tex.

The generator reads only completed, non-archived evaluation summaries.  Its
outputs are review artefacts; the dissertation remains self-contained because
the verified table blocks are inlined into thesis.tex.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = RESULTS / "evaluation" / "thesis_tables"

COND = {
    "periodic_theta89": "Periodic $89^\\circ$",
    "pml_outside_theta45": "PML $45^\\circ$",
}
METHOD = {
    "Fourier inverse": "Fourier inverse",
    "Time reversal": "\\kwave{} time reversal",
    "Iterated time reversal": "Iterated time reversal",
    "Gradient descent (1/L)": "\\kwave{} gradient descent",
    "FNO-only": "FNO-only optimisation",
    "Fourier-to-FNO": "Fourier-to-FNO optimisation",
    "FNO-to-Fourier": "FNO-to-Fourier optimisation",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of string-valued row dictionaries."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(name, text):
    """Write one generated LaTeX table without source indentation."""
    cleaned_lines = [
        line[4:] if line.startswith("    ") else line for line in text.strip().splitlines()
    ]
    output = "\n".join(cleaned_lines) + "\n"
    (OUT / f"{name}.tex").write_text(output, encoding="utf-8")


def format_number(value: float | str, decimal_places: int = 4) -> str:
    """Format a table value in fixed-point or scientific notation."""
    value = float(value)
    if abs(value) < 0.01 and value != 0:
        return f"{value:.2e}"
    return f"{value:.{decimal_places}f}"


def latex_table(caption: str, label: str, body: str, size: str = "\\small") -> str:
    """Wrap a generated tabular body in a complete LaTeX table environment."""
    return f"""\\begin{{table}}[H]
\\centering
{size}
\\caption{{{caption}}}
\\label{{{label}}}
{body}
\\end{{table}}"""


def generate_method_taxonomy() -> None:
    """Generate the forward-operator taxonomy table."""
    # 1 Method taxonomy
    save(
        "01_method_taxonomy",
        latex_table(
            "Taxonomy of the evaluated forward operators and placement of learned corrections.",
            "tab:method-cost",
            r"""\resizebox{\linewidth}{!}{%
    \begin{tabular}{lllll}
    \toprule
    Method & Learned domain & Analytical component & Trainable & Principal assumption \\
    \midrule
    $c\times$Fourier & none & FFT propagator & no & homogeneous periodic model \\
    FNO-only & complete map & none & yes & data cover the operator \\
    Fourier-to-FNO & measurement space & FFT input & yes & discrepancy is learnable \\
    FNO-to-Fourier & image space & differentiable FFT & yes & target lies near FFT range \\
    \kwave{} & none & time-domain solver & no & discretised acoustic model \\
    \bottomrule
    \end{tabular}}""",
        ),
    )


def generate_acquisition_table() -> None:
    """Generate the physical acquisition and dataset table."""
    save(
        "02_acquisition",
        latex_table(
            "Physical grid, acquisition geometry and dataset sizes used by the completed experiments.",
            "tab:acquisition-config",
            r"""\begin{tabular}{lcc}
    \toprule
    Property & Periodic $89^\circ$ & PML $45^\circ$ \\
    \midrule
    Source grid & $64\times64$ & $64\times64$ \\
    Grid spacing $(d_x,d_y)$ & $(10^{-4},10^{-4})$ m & $(10^{-4},10^{-4})$ m \\
    Sound speed / density & $1500$ m s$^{-1}$ / $1000$ kg m$^{-3}$ & same \\
    CFL / time step & $1.0$ / $6.67\times10^{-8}$ s & same \\
    Sensors / time samples & $64$ / $91$ & $64$ / $91$ \\
    Maximum view angle & $89^\circ$ & $45^\circ$ \\
    Boundary treatment & periodic & exterior PML \\
    Medium train/validation/test & $5{,}000/1{,}000/1{,}000$ & same \\
    Large train/validation/test & $50{,}000/5{,}000/10{,}000$ & same \\
    \bottomrule
    \end{tabular}""",
        ),
    )


def generate_protocol_tables() -> None:
    """Generate the forward-training and reconstruction protocol tables."""
    save(
        "03_forward_protocol",
        latex_table(
            "Forward-operator training protocol. Values are read from the completed training histories.",
            "tab:forward-protocol",
            r"""\begin{tabular}{lcc}
    \toprule
    Setting & Medium & Large \\
    \midrule
    Completed epochs / batch size & $5/128$ & $1/256$ \\
    Fourier modes / width / layers & $8/16/3$ & $8/16/3$ \\
    Optimiser / initial learning rate & AdamW / $2\times10^{-3}$ & same \\
    Training seed & 20260728 & 20260728 \\
    \bottomrule
    \end{tabular}""",
        ),
    )

    save(
        "04_reconstruction_protocol",
        latex_table(
            "Reconstruction and subsampling protocol. Values are read from the completed reconstruction records.",
            "tab:reconstruction-protocol",
            r"""\begin{tabular}{ll}
    \toprule
    Setting & Reported value \\
    \midrule
    Test images / retention probabilities & $1{,}000$ / $\{0.10,0.25,0.50,1.00\}$ \\
    Mask seed / bootstrap resamples & 20260802 / $2{,}000$ \\
    GD iterations / power iterations & $20$ (robustness), $80$ (detailed 25\%) / $8$ \\
    Iterated-TR iterations & $20$ (robustness), $80$ (detailed 25\%) \\
    Iterated-TR step at 25\%, periodic/PML & $1.5/2.0$ \\
    Iterated-TR step at 100\%, periodic/PML & $0.75/1.75$ \\
    Learned optimisation & $200$ Adam steps, learning rate $3\times10^{-2}$ \\
    Image constraint & projection to $[0,1]$ \\
    \bottomrule
    \end{tabular}""",
        ),
    )


def generate_forward_accuracy_table() -> None:
    """Generate the held-out forward-accuracy table."""
    blocks = []
    for scale, path, _n in [
        ("Medium", RESULTS / "mnist_medium/mnist_medium_v1/comparison.json", 1000),
        ("Large", RESULTS / "mnist_large/mnist_large_v1/comparison.json", 10000),
    ]:
        data = json.loads(path.read_text())["rows"]
        for r in data:
            blocks.append(
                (
                    scale,
                    r["condition"],
                    r["model"],
                    r["rel_l2_mean"],
                    r["centered_corr_mean"],
                    r["mse"],
                )
            )
    lines = []
    for i, (scale, c, m, rl, co, ms) in enumerate(blocks):
        if i and (scale, c) != (blocks[i - 1][0], blocks[i - 1][1]):
            lines.append("\\addlinespace")
        name = {
            "c \u00d7 Fourier baseline": "$c\\times$Fourier",
            "Fourier \u2192 FNO": "Fourier-to-FNO",
            "FNO \u2192 Fourier": "FNO-to-Fourier",
        }.get(m, m)
        prefix = (
            f"{scale} & {COND[c]} & "
            if i == 0 or (scale, c) != (blocks[i - 1][0], blocks[i - 1][1])
            else "& & "
        )
        best = name == "Fourier-to-FNO"
        vals = [format_number(rl), format_number(co), f"{float(ms):.6f}"]
        if best:
            vals = [f"\\textbf{{{v}}}" for v in vals]
        lines.append(prefix + name + " & " + " & ".join(vals) + r" \\")
    save(
        "04_forward_accuracy",
        latex_table(
            "Held-out forward accuracy. Medium and large test sets contain 1,000 and 10,000 images, respectively. Bold denotes the best method within each scale and condition; each learned checkpoint is from one completed training seed.",
            "tab:forward-results",
            """\\setlength{\\tabcolsep}{3.2pt}
    \\begin{tabular}{lllrrr}
    \\toprule
    Scale & Condition & Method & Rel. $L_2$ $\\downarrow$ & Corr. $\\uparrow$ & MSE $\\downarrow$ \\\\
    \\midrule
    """
            + "\n".join(lines)
            + """
    \\bottomrule
    \\end{tabular}""",
        ),
    )


def generate_sample_efficiency_table() -> None:
    """Generate the sample-efficiency table."""
    se = read_csv_rows(
        RESULTS / "evaluation/mnist_large_v1/required_experiments/sample_efficiency_summary.csv"
    )
    lookup = {(r["condition"], r["model"], int(r["train_samples"])): r for r in se}
    lines = []
    sizes = [1000, 5000, 10000, 25000, 50000]
    for c in COND:
        for j, m in enumerate(("FNO-only", "Fourier-to-FNO")):
            group = [lookup[c, m, n] for n in sizes]
            vals = [
                f"{float(r['rel_l2_mean']):.3f} $\\pm$ {float(r['rel_l2_std_across_seeds']):.3f}"
                for r in group
            ]
            needed = next(
                (n for n, r in zip(sizes, group, strict=False) if float(r["rel_l2_mean"]) <= 0.22),
                None,
            )
            lines.append(
                (COND[c] if j == 0 else "")
                + " & "
                + m
                + " & "
                + " & ".join(vals)
                + f" & {needed // 1000}k \\\\"
            )
    save(
        "05_sample_efficiency",
        latex_table(
            "Forward relative $L_2$ error across five training-set sizes, reported as mean $\\pm$ SD over three independent training seeds. The final column gives the first tested size whose across-seed mean is at most 0.22; the threshold is descriptive rather than a universal sample-complexity constant.",
            "tab:efficiency-final",
            r"""\resizebox{\linewidth}{!}{%
    \begin{tabular}{llrrrrrr}
    \toprule
    Condition & Model & 1k & 5k & 10k & 25k & 50k & $\leq0.22$ \\
    \midrule
    """
            + "\n".join(lines)
            + r"""
    \bottomrule
    \end{tabular}}""",
            "\\scriptsize",
        ),
    )


def generate_runtime_table() -> None:
    """Generate the forward-runtime table."""
    sources = {
        "CPU, batch 1": ("cpu", 1),
        "RTX 4090, batch 1": ("cuda_batch1", 1),
        "RTX 4090, batch 64": ("cuda_batch64", 64),
    }
    rt = {}
    for mode, (suffix, _batch) in sources.items():
        for c in COND:
            for r in read_csv_rows(RESULTS / f"evaluation/mnist_large_v1/runtime/{c}_{suffix}.csv"):
                rt[mode, c, r["method"]] = r
    order = ["Fourier", "fno_only", "fourier_to_fno", "fno_to_fourier", "k-Wave"]
    disp = {
        "fno_only": "FNO-only",
        "fourier_to_fno": "Fourier-to-FNO",
        "fno_to_fourier": "FNO-to-Fourier",
    }
    lines = []
    for mode, (_suffix, batch) in sources.items():
        for j, m in enumerate(order if mode == "CPU, batch 1" else order[:-1]):
            x, y = rt[mode, "periodic_theta89", m], rt[mode, "pml_outside_theta45", m]
            pa = f"{float(x['mean_ms_per_sample']):.3f} $\\pm$ {float(x['std_ms_per_sample']):.3f}"
            pb = f"{float(y['mean_ms_per_sample']):.3f} $\\pm$ {float(y['std_ms_per_sample']):.3f}"
            sp = (
                "$1\\times$"
                if m == "k-Wave"
                else f"{float(x['speedup_vs_kwave']):.1f}--{float(y['speedup_vs_kwave']):.1f}$\\times$"
            )
            lines.append(
                (mode if j == 0 else "")
                + f" & {batch} & {disp.get(m, m)} & {pa} & {pb} & {sp} \\\\"
            )
        lines.append("\\addlinespace")
    save(
        "06_runtime",
        latex_table(
            "Forward runtime in milliseconds per sample. CPU and RTX 4090 batch-one measurements use 200 repetitions after 20 warm-up calls; RTX 4090 batch-64 measurements use 50 repetitions after 10 warm-up calls. CPU batch-one speed-ups are hardware controlled. GPU speed-ups use the contemporaneously measured CPU \\kwave{} reference and therefore describe the implemented system rather than intrinsic algorithmic acceleration. The three learned models each contain 100,337 trainable parameters and occupy approximately 783 kB.",
            "tab:runtime-final",
            r"""\resizebox{\linewidth}{!}{%
    \begin{tabular}{lllrcc}
    \toprule
    Execution mode & Batch & Method & Periodic ms & PML ms & Speed-up \\
    \midrule
    """
            + "\n".join(lines[:-1])
            + r"""
    \bottomrule
    \end{tabular}}""",
            "\\scriptsize",
        ),
    )


def generate_itr_selection_table() -> None:
    """Generate the validation-selected ITR step-size table."""
    itrroot = RESULTS / "reconstruction/mnist_medium_v1/iterated_time_reversal"
    spec = [
        ("periodic_theta89", ".25", "0.5, 1, 1.5, 2, 2.5", "1.5", 100, 80),
        ("pml_outside_theta45", ".25", "0.5, 1, 1.5, 2, 2.5", "2", 100, 80),
        ("periodic_theta89", "1.00", "0.1, 0.25, 0.5, 0.75, 1", "0.75", 50, 20),
        ("pml_outside_theta45", "1.00", "0.1--2.5", "1.75", 50, 20),
    ]
    lines = []
    for c, k, cands, step, n, _it in spec:
        keep = "0.25" if k == ".25" else "1.00"
        p = itrroot / f"{c}_validation_keep{keep}_seed20260802_step{step}_metrics.json"
        d = json.loads(p.read_text())
        ret = f"{float(keep):.0%}".replace("%", r"\%")
        lines.append(
            f"{COND[c]} & {ret} & {cands} & {step} & {d['final_relative_l2_mean']:.4f} & {n} \\\\"
        )
    save(
        "07_itr_selection",
        latex_table(
            "Validation-only selection of the iterated time-reversal step. The selected value is subsequently frozen for test evaluation; $n$ denotes validation images.",
            "tab:itr-selection",
            r"""\begin{tabular}{lclrrr}
    \toprule
    Condition & Retention & Candidate steps & Selected & Val. rel. $L_2$ & $n$ \\
    \midrule
    """
            + "\n".join(lines)
            + r"""
    \bottomrule
    \end{tabular}""",
        ),
    )


def load_reconstruction_uncertainty() -> dict[tuple[str, float, str], dict[str, str]]:
    """Load reconstruction uncertainty rows indexed by experiment identity."""
    unc = read_csv_rows(
        RESULTS
        / "evaluation/mnist_medium_v1/required_experiments/reconstruction_with_uncertainty.csv"
    )
    return {(r["condition"], float(r["keep_fraction"]), r["method"]): r for r in unc}


def generate_reconstruction_25_table(
    uncertainty: dict[tuple[str, float, str], dict[str, str]],
) -> None:
    """Generate the detailed 25% retention reconstruction table."""
    U = uncertainty
    lines = []
    for c in COND:
        for j, m in enumerate(METHOD):
            r = U[c, 0.25, m]
            rel = f"{float(r['relative_l2_mean']):.4f} [{float(r['relative_l2_ci95_low']):.4f}, {float(r['relative_l2_ci95_high']):.4f}]"
            vals = [rel, format_number(r["correlation_mean"]), format_number(r["mse_mean"], 5)]
            isbest = m == "Gradient descent (1/L)"
            if isbest:
                vals = [f"\\textbf{{{x}}}" for x in vals]
            lines.append(
                (COND[c] if j == 0 else "") + " & " + METHOD[m] + " & " + " & ".join(vals) + r" \\"
            )
        if c == "periodic_theta89":
            lines.append("\\addlinespace")
    save(
        "08_reconstruction_25",
        latex_table(
            "Reconstruction at 25\\% measurement retention on 1,000 test images. Relative $L_2$ is reported as mean [bootstrap 95\\% CI] using 2,000 resamples; correlation and MSE are means. Bold denotes the best value within each condition.",
            "tab:reconstruction-results",
            r"""\resizebox{\linewidth}{!}{%
    \begin{tabular}{llccc}
    \toprule
    Condition & Method & Rel. $L_2$ [95\% CI] $\downarrow$ & Corr. $\uparrow$ & MSE $\downarrow$ \\
    \midrule
    """
            + "\n".join(lines)
            + r"""
    \bottomrule
    \end{tabular}}""",
        ),
    )


def generate_reconstruction_robustness_table(
    uncertainty: dict[tuple[str, float, str], dict[str, str]],
) -> None:
    """Generate the reconstruction robustness table."""
    U = uncertainty
    lines = []
    for c in COND:
        for j, m in enumerate(METHOD):
            vals = []
            for k in (0.1, 0.25, 0.5, 1.0):
                r = U[c, k, m]
                vals.append(
                    f"{float(r['relative_l2_mean']):.3f} [{float(r['relative_l2_ci95_low']):.3f}, {float(r['relative_l2_ci95_high']):.3f}]"
                )
            lines.append(
                (COND[c] if j == 0 else "") + " & " + METHOD[m] + " & " + " & ".join(vals) + r" \\"
            )
        if c == "periodic_theta89":
            lines.append("\\addlinespace")
    save(
        "09_robustness",
        latex_table(
            "Reconstruction robustness across retained measurements. Entries are mean relative $L_2$ [bootstrap 95\\% CI] over 1,000 test images and 2,000 bootstrap resamples.",
            "tab:retention-robustness",
            r"""\resizebox{\linewidth}{!}{%
    \begin{tabular}{llcccc}
    \toprule
    Condition & Method & 10\% & 25\% & 50\% & 100\% \\
    \midrule
    """
            + "\n".join(lines)
            + r"""
    \bottomrule
    \end{tabular}}""",
            "\\scriptsize",
        ),
    )


def generate_convergence_diagnostic_tables() -> None:
    """Generate the Lipschitz and finite-iteration diagnostic tables."""
    lip = read_csv_rows(
        RESULTS / "evaluation/mnist_medium_v1/required_experiments/lipschitz_step_size_summary.csv"
    )
    liplines = []
    for r in lip:
        ret = f"{float(r['keep_fraction']):.0%}".replace("%", r"\%")
        liplines.append(
            f"{COND[r['condition']]} & {ret} & {float(r['lipschitz_mean']):.3f} $\\pm$ {float(r['lipschitz_std']):.3f} & {float(r['step_size_mean']):.3f} $\\pm$ {float(r['step_size_std']):.3f} \\\\"
        )
    conv = read_csv_rows(
        RESULTS / "evaluation/mnist_medium_v1/required_experiments/convergence_fits.csv"
    )
    convlines = []
    for r in conv:
        if abs(float(r["keep_fraction"]) - 0.25) > 1e-9:
            continue
        metric = "rel. $L_2$" if r["metric"] == "rel-L2" else r["metric"]
        convlines.append(
            f"{COND[r['condition']]} & {r['method']} & {metric} & "
            f"{float(r['semilog_slope']):.4f} & "
            f"{float(r['semilog_r2']):.3f} & "
            f"{float(r['loglog_slope']):.4f} & "
            f"{float(r['loglog_r2']):.3f}" + r" \\"
        )
    save(
        "11_lipschitz_steps",
        latex_table(
            "Power-iteration estimates of the Lipschitz constant and the resulting gradient-descent step. Values are mean $\\pm$ SD over 1,000 test images.",
            "tab:lipschitz-steps",
            r"""\begin{tabular}{lccc}
    \toprule
    Condition & Retention & Estimated $L$ & Step $1/L$ \\
    \midrule
    """
            + "\n".join(liplines)
            + r"""
    \bottomrule
    \end{tabular}""",
        ),
    )

    save(
        "12_convergence_fits",
        latex_table(
            "Descriptive semilog and log--log regression fits at 25\\% retention. The finite-iteration slopes are diagnostics rather than theoretical convergence orders.",
            "tab:convergence-fits",
            r"""\resizebox{\linewidth}{!}{%
    \begin{tabular}{lllrrrr}
    \toprule
    Condition & Method & Metric & Semilog slope & $R^2$ & Log--log slope & $R^2$ \\
    \midrule
    """
            + "\n".join(convlines)
            + r"""
    \bottomrule
    \end{tabular}}""",
            "\\scriptsize",
        ),
    )


def write_manifest() -> None:
    """Record the authoritative result sources used by the table generator."""
    manifest = {
        "authoritative_sources": [
            "results/mnist_medium/mnist_medium_v1/comparison.json",
            "results/mnist_large/mnist_large_v1/comparison.json",
            "results/evaluation/mnist_large_v1/required_experiments/sample_efficiency_summary.csv",
            "results/evaluation/mnist_large_v1/runtime/*_cuda.csv",
            "results/evaluation/mnist_medium_v1/required_experiments/reconstruction_with_uncertainty.csv",
            "results/evaluation/mnist_medium_v1/required_experiments/lipschitz_step_size_summary.csv",
            "results/evaluation/mnist_medium_v1/required_experiments/convergence_fits.csv",
            "results/reconstruction/mnist_medium_v1/iterated_time_reversal/*validation*_metrics.json",
        ],
        "excluded": "archive_pre_robustness and smoke outputs",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    """Generate every dissertation table from saved summaries."""
    OUT.mkdir(parents=True, exist_ok=True)
    generate_method_taxonomy()
    generate_acquisition_table()
    generate_protocol_tables()
    generate_forward_accuracy_table()
    generate_sample_efficiency_table()
    generate_runtime_table()
    generate_itr_selection_table()
    uncertainty = load_reconstruction_uncertainty()
    generate_reconstruction_25_table(uncertainty)
    generate_reconstruction_robustness_table(uncertainty)
    generate_convergence_diagnostic_tables()
    write_manifest()
    print(f"Saved 12 table blocks in {OUT}")


if __name__ == "__main__":
    main()

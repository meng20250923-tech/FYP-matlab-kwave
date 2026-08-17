# Reproducibility example

This directory contains the fixed inputs required to run a small result from the
thesis code without downloading MNIST or generating new k-Wave data.

The example evaluates four samples from the large periodic test split. It
recomputes the analytical Fourier prediction, evaluates the supplied FNO-only
checkpoint, applies the saved 25% measurement masks, and reconstructs the images
with the analytical Fourier inverse. The resulting metrics are compared with the
stored expected values.

Run the example from the repository root:

```bash
python -m scripts.smoke.run_reproducibility_check \
  --config smoke/reproducibility.yaml
```

The generated summary is written to `results/smoke/summary.json`. This compact
example verifies the software and model interfaces. It does not replace the
medium- and large-scale experiments reported in the thesis.

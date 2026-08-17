# Expected results

`periodic_example_metrics.json` contains deterministic metrics computed from the
fixed sample batch and supplied checkpoint. The reproducibility command compares
newly evaluated values with these references using the tolerances recorded in
`configs/smoke/reproducibility.yaml`.

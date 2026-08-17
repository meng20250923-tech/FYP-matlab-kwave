"""Dataset generation, preprocessing, and storage utilities."""

from pat_fno.data.mnist import (
    build_setting,
    conditions,
    generate_sample,
    load_arrays,
    load_mnist_splits,
    save_json,
    shard_paths,
    to_pressure,
    write_shard,
)

__all__ = [
    "build_setting",
    "conditions",
    "generate_sample",
    "load_arrays",
    "load_mnist_splits",
    "save_json",
    "shard_paths",
    "to_pressure",
    "write_shard",
]

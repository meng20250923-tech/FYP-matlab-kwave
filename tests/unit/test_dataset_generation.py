"""Regression tests for deterministic dataset-generation orchestration."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts.data import generate_mnist_dataset as generation


class RecordingPool:
    """Return prepared results while recording ordered imap calls."""

    def __init__(self):
        self.calls = []

    def imap(self, function, images, chunksize):
        images = list(images)
        self.calls.append((function, chunksize, [int(image[0, 0]) for image in images]))
        return [
            tuple(
                np.full((2, 3), int(image[0, 0]) + offset, dtype=np.float64) for offset in range(4)
            )
            for image in images
        ]


class ContextPool(RecordingPool):
    """Context-managed pool used to verify construction arguments."""

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class RecordingContext:
    """Record multiprocessing Pool arguments."""

    def __init__(self):
        self.pool = ContextPool()
        self.arguments = []

    def Pool(self, processes, initializer, initargs, maxtasksperchild):
        self.arguments.append((processes, initializer, initargs, maxtasksperchild))
        return self.pool


def make_setting(boundary="periodic", theta=89.0):
    """Construct the subset of a physical setting needed by the generator."""
    return SimpleNamespace(
        kwaveBoundary=boundary,
        Nt=91,
        dt=1.0 / 15_000_000,
        soundSpeed=1500.0,
        computation=SimpleNamespace(theta_max=np.deg2rad(theta)),
    )


def test_generate_split_shards_preserves_order_and_boundaries(tmp_path, monkeypatch):
    """Ordered samples must be written using unchanged shard boundaries."""
    pool = RecordingPool()
    images = np.stack([np.full((2, 2), value, dtype=np.uint8) for value in range(1, 8)])
    labels = np.arange(7, dtype=np.int64)
    indices = np.arange(100, 107, dtype=np.int64)
    writes = []

    def record_write(path, p0, raw, scaled, kwave, shard_labels, shard_indices, attributes):
        writes.append(
            (
                path.name,
                p0.copy(),
                raw.copy(),
                scaled.copy(),
                kwave.copy(),
                shard_labels.copy(),
                shard_indices.copy(),
                attributes,
            )
        )

    monkeypatch.setattr(generation, "write_shard", record_write)
    generation.generate_split_shards(
        pool,
        tmp_path,
        "periodic_theta89",
        "train",
        (images, labels, indices),
        shard_size=3,
        workers=2,
        setting=make_setting(),
    )

    assert [write[0] for write in writes] == [
        "train_00000_00003.h5",
        "train_00003_00006.h5",
        "train_00006_00007.h5",
    ]
    assert [call[1:] for call in pool.calls] == [
        (1, [1, 2, 3]),
        (1, [4, 5, 6]),
        (1, [7]),
    ]
    assert np.array_equal(np.concatenate([write[5] for write in writes]), labels)
    assert np.array_equal(np.concatenate([write[6] for write in writes]), indices)
    assert np.array_equal(np.concatenate([write[1][:, 0, 0] for write in writes]), images[:, 0, 0])
    assert all(write[1].dtype == np.float32 for write in writes)


def test_generate_condition_keeps_pool_and_split_order(tmp_path, monkeypatch):
    """Each condition must use one configured pool and ordered split traversal."""
    context = RecordingContext()
    setting = make_setting()
    split_arrays = (np.zeros((1, 2, 2)), np.zeros(1), np.zeros(1))
    raw_splits = {name: split_arrays for name in ("train", "validation", "test")}
    args = SimpleNamespace(workers=4, maxtasksperchild=11, split="all")
    manifest = {"conditions": {}}
    calls = []
    saved = []
    monkeypatch.setattr(generation, "build_setting", lambda _condition: setting)
    monkeypatch.setattr(
        generation,
        "generate_split_shards",
        lambda pool,
        output,
        condition,
        split,
        arrays,
        shard_size,
        workers,
        current_setting: calls.append(
            (pool, output, condition, split, arrays, shard_size, workers, current_setting)
        ),
    )
    monkeypatch.setattr(
        generation, "save_json", lambda path, content: saved.append((path, content.copy()))
    )

    generation.generate_condition(
        context,
        args,
        tmp_path,
        raw_splits,
        ("train", "validation", "test"),
        250,
        "periodic_theta89",
        manifest,
        tmp_path / "manifest.json",
    )

    assert context.arguments == [(4, generation._worker_init, ("periodic_theta89",), 11)]
    assert [call[3] for call in calls] == ["train", "validation", "test"]
    assert all(call[0] is context.pool for call in calls)
    assert manifest["conditions"]["periodic_theta89"]["complete"] is True
    assert len(saved) == 1


def test_manifest_initialisation_preserves_completed_conditions(tmp_path):
    """Resumed generation must retain condition records from existing manifests."""
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        '{"conditions":{"periodic_theta89":{"complete":true}},"complete":true}'
    )

    returned_path, manifest = generation.initialise_manifest(
        tmp_path,
        "mnist_large_v1",
        "large",
        {"train": 50_000, "validation": 5_000, "test": 10_000},
    )

    assert returned_path == manifest_path
    assert manifest["conditions"]["periodic_theta89"] == {"complete": True}
    assert manifest["dataset"] == "mnist_large_v1"
    assert manifest["scale"] == "large"
    assert manifest["generator"] == "parallel"

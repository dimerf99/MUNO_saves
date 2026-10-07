from muno.data.benchmarks.pipeline import build_benchmark_loaders


ce_crp_config = {
    "source": {
        "location": "huggingface",
        "format": "netcdf",
        "repo_id": "camlab-ethz/CE-CRP",
        "filename": "data_0.nc",
        "cache_dir": r"D:\datasets_cache",
        "variable_name": "data",
        "sample_dim": "sample",
    },
    "adapter": {
        "type": "poseidon_temporal",
        "variable_name": "data",
        "input_time_indices": (0, 1, 2),
        "output_time_indices": (3, 4),
        "input_channel_indices": (0, 1, 2, 3, 4),
        "output_channel_indices": (0, 1, 2, 3, 4),
        "physics_name": "CE-CRP",
    },
    "split": {
        "train": (0, 32),
        "val": (32, 48),
        "test": (48, 64),
    },
    "max_samples_per_split": {
        "train": 16,
        "val": 16,
        "test": 16,
    },
    "loaders": {
        "train": {
            "batch_size": 16,
            "num_workers": 0,
            "pin_memory": False,
            "shuffle": True,
            "drop_last": True,
        },
        "val": {
            "batch_size": 16,
            "num_workers": 0,
            "pin_memory": False,
            "shuffle": False,
            "drop_last": False,
        },
        "test": {
            "batch_size": 16,
            "num_workers": 0,
            "pin_memory": False,
            "shuffle": False,
            "drop_last": False,
        },
    },
}

train_loader, val_loader, test_loader = build_benchmark_loaders(ce_crp_config)

print(len(train_loader))
print(len(val_loader))
print(len(test_loader))

batch = next(iter(train_loader))
print(batch.keys())
print(batch["x"].shape)
print(batch["y"].shape)
print(batch["benchmark_name"])
print(batch["physics_name"])
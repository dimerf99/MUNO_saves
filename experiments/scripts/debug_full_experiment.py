from muno.data.benchmarks.pipeline import build_benchmark_loaders

se_af_config = {
    # # LOCAL SE-AF
    # "source": {
    #     "location": "local",
    #     "format": "netcdf",
    #     "path": r"D:\PycharmProjects\DeepLearningBalakirev\data\SE-AF\SE-AF.nc",
    #     "variable_name": "solution",
    #     "sample_dim": "sample",
    # },
    # # HF SE-AF
    # "source": {
    #     "location": "huggingface",
    #     "format": "netcdf",
    #     "repo_id": "camlab-ethz/SE-AF",
    #     "filename": "SE-AF.nc",
    #     "cache_dir": r"D:\datasets_cache",
    #     "variable_name": "solution",
    #     "sample_dim": "sample",
    # },
    # HF Wave-Gauss
    "source": {
        "location": "huggingface",
        "format": "netcdf",
        "repo_id": "camlab-ethz/Wave-Gauss",
        "filename": "SE-AF.nc",
        "cache_dir": r"D:\datasets_cache",
        "variable_name": "solution",
        "sample_dim": "sample",
    },
    "adapter": {
        "type": "se_af",
        "input_indices": (0, 1, 2, 3),
        "output_indices": (4, 5),
        "include_c": True,
        "benchmark_name": "POSEIDON",
        "physics_name": "Wave-Gauss"
    },
    "split": {
        "train": (0, 10509),
        "val": (10509, 10629),
        "test": (10629, 10869),
    },
    "max_samples_per_split": {
        "train": 32,
        "val": 16,
        "test": 16,
    },
    "loaders": {
        "train": {
            "batch_size": 16,
            "num_workers": 0,
            "pin_memory": False,
            "shuffle": True,
            "drop_last": True
        },
        "val": {
            "batch_size": 16,
            "num_workers": 0,
            "pin_memory": False,
            "shuffle": False,
            "drop_last": False
        },
        "test": {
            "batch_size": 16,
            "num_workers": 0,
            "pin_memory": False,
            "shuffle": False,
            "drop_last": False
        },
    },
}

train_loader, val_loader, test_loader = build_benchmark_loaders(se_af_config)

print(len(train_loader))
print(len(val_loader))
print(len(test_loader))

batch = next(iter(train_loader))
print(batch.keys())
print(batch["x"].shape)
print(batch["y"].shape)
print(batch["benchmark_name"])
print(batch["physics_name"])

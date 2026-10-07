from muno.data.benchmarks.pipeline import build_benchmark_loaders

# source:
#   location: local        # local | hf
#   format: netcdf         # netcdf | hdf5 | arrow | zarr
#   path: ...
#   repo_id: camlab-ethz/SE-AF
#   filename: SE-AF.nc
#   cache_dir: ...
#   variable_name: solution
#   sample_dim: sample

se_af_config = {
    "path": "D:\\PycharmProjects\\DeepLearningBalakirev\\data\\SE-AF\\SE-AF.nc",
    "source_type": "netcdf",
    "adapter_type": "se_af",
    "variable_name": "solution",
    "sample_dim": "sample",
    "input_indices": (0,),
    "output_indices": (1,),
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
    "train_loader": {
        "batch_size": 16,
        "num_workers": 0,
        "pin_memory": False,
        "shuffle": True,
        "drop_last": True,
    },
    "val_loader": {
        "batch_size": 16,
        "num_workers": 0,
        "pin_memory": False,
        "shuffle": False,
        "drop_last": False,
    },
    "test_loader": {
        "batch_size": 16,
        "num_workers": 0,
        "pin_memory": False,
        "shuffle": False,
        "drop_last": False,
    }
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

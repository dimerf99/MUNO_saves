import torch

from muno.data.benchmarks.sources import DictSource, NetCDFSource, MultiNetCDFSource
from muno.data.benchmarks.datasets import LazyCanonicalDataset

from muno.data.benchmarks.pipeline import build_adapter

# components = {
#     "solution": DictSource({"value": torch.randn(10, 15, 128, 128)}),
#     "c": DictSource({"value": torch.randn(10, 128, 128)}),
# }
#
# source = MultiNetCDFSource(components)
#
# sample = source.get_sample(0)
#
# solution = sample["solution"]  # [15, 128, 128]
# c = sample["c"]         # [128, 128]
#
# data = {
#     "solution": torch.randn(10, 15, 128, 128),
#     "c": torch.randn(10, 128, 128),
# }
#
# source = DictSource(data)
# print(len(source))
#
# adapter_config = {
#     "type": "temporal",
#     "variable_name": "solution",
#     "data_order": "THW",
#     "input_time_indices": (0, 1, 2, 3),
#     "output_time_indices": (4, 5),
#     "static_inputs": [{"variable_name": "c", "data_order": "HW", "target": "x"}],
#     "benchmark_name": "POSEIDON",
#     "physics_name": "Wave-Gauss",
# }
#
# adapter = build_adapter(adapter_config)
#
# dataset = LazyCanonicalDataset(source, adapter)
# item = dataset[0]
# print(item.keys())
# print(item["x"].shape)
# print(item["y"].shape)
#
# loader_1 = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, drop_last=False)
# print(f"\nbatch_size=1")
# print(len(loader_1))
#
# batch = next(iter(loader_1))
# print(batch["x"].shape)
# print(batch["y"].shape)
#
# loader_2 = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, drop_last=False)
# print(f"\nbatch_size=2")
# print(len(loader_2))
#
# batch = next(iter(loader_2))
# print(batch["x"].shape)
# print(batch["y"].shape)

from muno.data.benchmarks.pipeline import build_benchmark_loaders

wave_gauss_config = {
    "source": {
        "location": "huggingface",
        "format": "multi_netcdf",
        "repo_id": "camlab-ethz/Wave-Gauss",
        "cache_dir": r"D:\datasets_cache",
        "length_component": "solution",
        "components": {
            "solution": {
                "filename": "solution_0.nc",
                "variable_name": "solution",
                "sample_dim": "sample",
            },
            "c": {
                "filename": "c_0.nc",
                "variable_name": "c",
                "sample_dim": "sample",
            },
        },
    },
    "adapter": {
        "type": "wave_gauss",
        "input_time_indices": (0, 1, 2, 3),
        "output_time_indices": (4, 5),
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

train_loader, val_loader, test_loader = build_benchmark_loaders(wave_gauss_config)

print(len(train_loader))
print(len(val_loader))
print(len(test_loader))

batch = next(iter(train_loader))
print(batch.keys())
print(batch["x"].shape)
print(batch["y"].shape)
print(batch["benchmark_name"])
print(batch["physics_name"])

pass

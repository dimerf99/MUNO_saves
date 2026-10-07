import torch

from muno.data.benchmarks.sources import DictSource
from muno.data.benchmarks.datasets import LazyCanonicalDataset
from muno.data.benchmarks.pipeline import build_adapter


data = {
    "data": torch.randn(10, 21, 5, 128, 128),
}

source = DictSource(data)

adapter_config = {
    "type": "poseidon_temporal",
    "variable_name": "data",
    "input_time_indices": (0, 1, 2),
    "output_time_indices": (3, 4),
    "input_channel_indices": (0, 1, 2, 3, 4),
    "output_channel_indices": (0, 1, 2, 3, 4),
    "physics_name": "CE-CRP",
}

adapter = build_adapter(adapter_config)

dataset = LazyCanonicalDataset(source, adapter)
item = dataset[0]

print(item.keys())
print(item["x"].shape)
print(item["y"].shape)
print(item["benchmark_name"])
print(item["physics_name"])

loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, drop_last=False)
batch = next(iter(loader))

print(batch["x"].shape)
print(batch["y"].shape)

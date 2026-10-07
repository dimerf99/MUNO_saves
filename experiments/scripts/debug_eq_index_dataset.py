import torch

from muno.data.benchmarks.sources import DictSource
from muno.data.benchmarks.datasets import LazyCanonicalDataset
from muno.data.benchmarks.multiphysics_loaders import EqIndexDataset

from muno.data.benchmarks.pipeline import (
    apply_max_samples,
    build_adapter,
    build_datasets,
    build_loaders,
    build_source,
)


class DummyAdapter:
    def canonize(self, sample):
        return {
            "x": sample["x"],
            "y": sample["y"],
            "benchmark_name": "POSEIDON",
            "physics_name": "dummy",
            "metadata": {},
        }


source = DictSource({
    "x": torch.randn(4, 1, 8, 8),
    "y": torch.randn(4, 1, 8, 8),
})

dataset = LazyCanonicalDataset(source, DummyAdapter())
dataset = EqIndexDataset(dataset, eq_idx=3)

item = dataset[0]

print(item.keys())
print(item["x"].shape)
print(item["y"].shape)
print(item["eq_idx"])

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    drop_last=False,
)

batch = next(iter(loader))

print(batch["x"].shape)
print(batch["y"].shape)
print(batch["eq_idx"])
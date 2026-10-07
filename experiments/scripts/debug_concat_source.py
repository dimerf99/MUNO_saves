import torch

from muno.data.benchmarks.sources import DictSource, ConcatSource


source_0 = DictSource({
    "data": torch.arange(0, 3).reshape(3, 1, 1, 1).float()
})

source_1 = DictSource({
    "data": torch.arange(3, 7).reshape(4, 1, 1, 1).float()
})

source_2 = DictSource({
    "data": torch.arange(7, 9).reshape(2, 1, 1, 1).float()
})

concat_source = ConcatSource([source_0, source_1, source_2])

print(len(concat_source))

for idx in range(len(concat_source)):
    sample = concat_source.get_sample(idx)
    print(idx, sample["data"].item())

print("last:", concat_source.get_sample(-1)["data"].item())

from functools import singledispatch


# @singledispatch
# def get_loaders_channels(loaders):
#     raise NotImplementedError('Calling get_loaders_channels of a default unimplemented type.')
#
#
# @get_loaders_channels.register
# def _(loaders: list):
#     assert all(
#         [isinstance(loader, DataLoader) for loader in loaders]), 'Loaders have to be a list of DataLoader objects.'
#     return [get_loader_channels(loader) for loader in loaders]
#
#
# @get_loaders_channels.register
# def _(loaders: DataLoader):
#     batch = next(iter(loaders))
#     assert isinstance(batch, dict), \
#         'loader has to return a dict with keys - multiphysics problems idx, values - dicts {"x": torch.Tensor, "y": ...}.'
#
#     # print('batch is ', batch)
#     # print('Shapes are: ', [(subbatch['eq_idx'], subbatch["x"].shape, subbatch["y"].shape) for subbatch in batch.values()])
#     return [(subbatch["x"].shape[1], subbatch["y"].shape[1]) for subbatch in batch.values()]


# @singledispatch
# def describe(obj):
#     return f"default handler: {type(obj)}"
#
#
# @describe.register(list)
# def _(obj):
#     return f"list handler: length={len(obj)}"
#
#
# @describe.register(dict)
# def _(obj):
#     return f"dict handler: keys={list(obj)}"
#
#
# @describe.register(int)
# def _(obj):
#     return f"int handler: value={obj}"
#
#
# items = [10, 20, 30]
# config = {"mode": "train", "epochs": 5}
# epoch = 3
# name = "experiment"
#
# print(describe(items))
# print(describe(config))
# print(describe(epoch))
# print(describe(name))


from pprint import pprint

data = {
    "model_params": {"learning_rate": 0.001, "batch_size": 32, "optimizer": "AdamW"},
    "dataset_splits": ["train", "val", "test"],
    "metrics": {"accuracy": 0.95, "loss": 0.12, "f1_score": 0.94}
}

print(data)
pprint(data, sort_dicts=True)


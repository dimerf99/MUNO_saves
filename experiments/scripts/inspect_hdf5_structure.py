import argparse
from pathlib import Path

import h5py


def print_hdf5_tree(path):
    path = Path(path)
    print(f"\nFILE: {path}")

    with h5py.File(path, "r") as file:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(
                    f"DATASET {name}: "
                    f"shape={obj.shape}, dtype={obj.dtype}, "
                    f"chunks={obj.chunks}, compression={obj.compression}"
                )
            else:
                print(f"GROUP   {name}")

        file.visititems(visitor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()

    for path in args.paths:
        print_hdf5_tree(path)


if __name__ == "__main__":
    main()

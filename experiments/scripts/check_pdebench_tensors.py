import h5py

path = r"D:/PDEBench_data/NS_Incom/ns_incom_inhom_2d_512-0.h5"

with h5py.File(path, "r") as f:
    print("keys:", list(f.keys()))

    dset = f["velocity"]
    print("velocity shape:", dset.shape)
    print("sample_dim 0 length:", dset.shape[0])

# python -c
# ("import h5py; "
#  "p='/workspace/data/PDEBench_data/2D_CFD/2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5';
#  f=h5py.File(p,'r');
#  print('keys:', list(f.keys()));
#  [print(k, f[k].shape, f[k].dtype) for k in f.keys() if hasattr(f[k], 'shape')];
#  f.close()")

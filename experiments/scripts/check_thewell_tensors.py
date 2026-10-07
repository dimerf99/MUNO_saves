import h5py

# path = r'D:\TheWell_data\active_matter\hf_cache\datasets--polymathic-ai--active_matter\snapshots\dc3e9135a75b4e5a0d979086219fc97fe668f7a3\data\train\active_matter_L_10.0_zeta_17.0_alpha_-4.0.hdf5'
# path = r'D:\TheWell_data\active_matter\hf_cache\datasets--polymathic-ai--active_matter\snapshots\dc3e9135a75b4e5a0d979086219fc97fe668f7a3\data\valid\active_matter_L_10.0_zeta_17.0_alpha_-4.0.hdf5'
path = r'D:\TheWell_data\active_matter\hf_cache\datasets--polymathic-ai--active_matter\snapshots\dc3e9135a75b4e5a0d979086219fc97fe668f7a3\data\test\active_matter_L_10.0_zeta_17.0_alpha_-4.0.hdf5'

f = h5py.File(path,'r')

print(list(f.keys()))
print(path)
print(list(f['t0_fields'].keys()))
print(list(f['t1_fields'].keys()))
print(list(f['t2_fields'].keys()))

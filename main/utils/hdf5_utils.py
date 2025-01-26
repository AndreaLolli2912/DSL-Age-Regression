import h5py
import numpy as np

def save_dict_to_hdf5(dictionary, file_path):
    try:
        with h5py.File(file_path, 'w') as hdf5_file:
            for key, value in dictionary.items():
                group = hdf5_file.create_group(key)
                for feature_name, feature_values in value.items():
                    group.create_dataset(feature_name, data=np.array(feature_values))
        print(f"Dictionary successfully saved to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving the dictionary: {e}")

def read_hdf5_to_dict(file_path):
    try:
        result = {}
        with h5py.File(file_path, 'r') as hdf5_file:
            for group_name in hdf5_file:
                group = hdf5_file[group_name]
                result[group_name] = {key: group[key][...] for key in group}
        print(f"Data successfully loaded from {file_path}")
        return result
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

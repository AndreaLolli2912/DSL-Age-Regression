from preprocessing_audio_features import get_weighted_mean_freq, get_summary_statistics
import numpy as np
from hdf5_utils import read_hdf5_to_dict
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def extract_statistics_mfcc_delta(X):
    mfcc = X['mfcc']
    delta_mfcc = X['delta_mfcc']
    delta2_mfcc = X['delta2_mfcc']

    extracted_mfcc = np.apply_along_axis(get_summary_statistics, 1, mfcc).flatten()
    extracted_delta_mfcc = np.apply_along_axis(get_summary_statistics, 1, delta_mfcc).flatten()
    extracted_delta2_mfcc = np.apply_along_axis(get_summary_statistics, 1, delta2_mfcc).flatten()

    extracted = np.concatenate((extracted_mfcc, extracted_delta_mfcc, extracted_delta2_mfcc))
    return extracted

def sort_dataFrame(data, df_extracted):
    keys = data.keys()
    keys = [int(num.split('.')[0]) - 1 for num in keys]
    df_extracted['Id'] = keys
    return df_extracted.sort_values(by=['Id'])

dev_data = read_hdf5_to_dict(file_path="main/data/audio_features_development.h5")
eval_data = read_hdf5_to_dict(file_path="main/data/audio_features_evaluation.h5")

extracted_dev = np.zeros((240))
for k in dev_data:
    aux = extract_statistics_mfcc_delta(dev_data[k])    
    extracted_dev = np.vstack((extracted_dev,aux))

extracted_dev = extracted_dev[1:]

extracted_eval = np.zeros((240))
for k in eval_data:
    aux = extract_statistics_mfcc_delta(eval_data[k])    
    extracted_eval = np.vstack((extracted_eval,aux))

extracted_eval = extracted_eval[1:]

n_components = 0.90
pipeline = make_pipeline(StandardScaler(), PCA(n_components=n_components))
pipeline.fit(extracted_dev)
extracted_pca_dev = pipeline.transform(extracted_dev)
extracted_pca_eval = pipeline.transform(extracted_eval)

columns = ['pca_mfcc_delta_'+ str(i) for i in range(pipeline[1].n_components_)]

df_dev_extracted = pd.DataFrame(extracted_pca_dev, columns=columns)
df_eval_extracted = pd.DataFrame(extracted_pca_eval, columns=columns)

sort_dataFrame(dev_data, df_dev_extracted).to_csv('main/data/mfcc_delta_data/dev_mfcc_delta.csv', index=False)
sort_dataFrame(eval_data, df_eval_extracted).to_csv('main/data/mfcc_delta_data/eval_mfcc_delta.csv', index=False)
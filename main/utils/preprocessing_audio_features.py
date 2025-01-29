import numpy as np
import scipy.stats as sps
import librosa
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def get_weighted_mean_freq(S, sr=22050, fmin=0.0, fmax=None):
    
    n_mels = S.shape[0]

    mel_bin_freqs = librosa.mel_frequencies(
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax if fmax is not None else sr/2
    )

    low_bin = (mel_bin_freqs >= 40) & (mel_bin_freqs <= 150)
    mid_low_bin = (mel_bin_freqs > 150) & (mel_bin_freqs <= 500)
    mid_bin = (mel_bin_freqs > 500) & (mel_bin_freqs <= 2000)
    mid_high_bin = (mel_bin_freqs > 2000) & (mel_bin_freqs <= 4000)
    high_bin = (mel_bin_freqs > 4000) & (mel_bin_freqs <= 8000)
    very_high_bin = (mel_bin_freqs > 8000) & (mel_bin_freqs <= 12000)


    low_s = S[low_bin, :]
    mid_low_s = S[mid_low_bin, :]   
    mid_s = S[mid_bin, :]
    mid_high_s = S[mid_high_bin, :]
    high_s = S[high_bin, :]   
    very_high_s = S[very_high_bin, :] 

    def safe_weighted_mean(freqs, spec):
        # Sum over the frequency dimension
        sum_power = np.sum(spec, axis=0)
        weighted_sum = np.sum(freqs[:, None] * spec, axis=0)
        
        mean_freq = np.divide(weighted_sum,
                              sum_power,
                              out=np.zeros_like(weighted_sum),
                              where=(sum_power != 0))
        return mean_freq

    # Calculate weighted mean frequency in each bin
    low_mean_freq = safe_weighted_mean(mel_bin_freqs[low_bin], low_s)
    mid_low_mean_freq = safe_weighted_mean(mel_bin_freqs[mid_low_bin], mid_low_s)
    mid_mean_freq = safe_weighted_mean(mel_bin_freqs[mid_bin], mid_s)
    mid_high_mean_freq = safe_weighted_mean(mel_bin_freqs[mid_high_bin], mid_high_s)
    high_mean_freq = safe_weighted_mean(mel_bin_freqs[high_bin], high_s)
    very_high_mean_freq = safe_weighted_mean(mel_bin_freqs[very_high_bin], very_high_s)

    return low_mean_freq, mid_low_mean_freq, mid_mean_freq, mid_high_mean_freq, high_mean_freq, very_high_mean_freq

def get_summary_statistics(X: np.ndarray):
    mean = np.mean(X)

    std = np.std(X)
    kurtosis = sps.kurtosis(X)
    skewness = sps.skew(X)

    return mean, std, kurtosis, skewness

def compute_audio_feature_stats(audio_data):
    summary_statistics = ["mean", "std", "kurtosis", "skew"]
    freq_bands_means = [
        "low_mean_freq",
        "mid_low_mean_freq",
        "mid_mean_freq",
        "mid_high_mean_freq",
        "high_mean_freq",
        "very_high_mean_freq"
    ]

    # Dictionaries to hold aggregated results
    spectral_dict = {"file_name": []}
    melspectrogram_dict = {"file_name": []}
    mfcc_deltas_dict = {"file_name": []}
    spectral_contrast_dict = {"file_name": []}
    fundamental_freq_dict = {"file_name": []}

    # Collect them in a list to simplify adding file names
    results = [spectral_dict, melspectrogram_dict, mfcc_deltas_dict, spectral_contrast_dict, fundamental_freq_dict]

    # -------------------------------------------------------
    # 1) Iterate over each audio file
    # -------------------------------------------------------
    for file_name, file_data in audio_data.items():
        
        # Append file name to each results dictionary
        for dictionary in results:
            dictionary["file_name"].append(file_name)

        # ---------------------------------------------------
        # 2) Process each feature in the file
        # ---------------------------------------------------
        for feature_name, feature_data in file_data.items():

            # ---------------------------
            # A) Fundamental Freequency (m, )
            # ---------------------------
            if feature_name == "f0":
                stats_values = get_summary_statistics(feature_data.flatten())
                # Create dict: e.g. {"f0_mean": val, "f0_std": val, ...}
                summary_statistics_f0 = ["mean", "std", "kurtosis", "skew", "median", "min", "max"]
                stats_values = tuple(list(stats_values) + [np.median(feature_data), np.min(feature_data), np.max(feature_data)])
                statistical_summary = {
                    f"fundamental_frequency_{stat}": val
                    for stat, val in zip(summary_statistics_f0, stats_values)
                }
                # Store in spectral_dict
                for key, val in statistical_summary.items():
                    if key not in fundamental_freq_dict:
                        fundamental_freq_dict[key] = [val]
                    else:
                        fundamental_freq_dict[key].append(val)
            # ---------------------------
            # B) SPECTRAL DATA & RMS  (1, n)
            # ---------------------------
            elif feature_data.shape[0] == 1:
                stats_values = get_summary_statistics(feature_data.flatten())
                # Create dict: e.g. {"spectral_rms_mean": val, "spectral_rms_std": val, ...}
                statistical_summary = {
                    f"{feature_name}_{stat}": val
                    for stat, val in zip(summary_statistics, stats_values)
                }
                # Store in spectral_dict
                for key, val in statistical_summary.items():
                    if key not in spectral_dict:
                        spectral_dict[key] = [val]
                    else:
                        spectral_dict[key].append(val)

            # ---------------------------
            # B) MELSPECTROGRAM (128, n)
            # ---------------------------
            elif feature_data.shape[0] == 128:
                weighted_means = get_weighted_mean_freq(feature_data)
                # Create dict: e.g. {"melspectrogram_low_mean_freq": val, ...}
                freq_bands_data = {
                    f"{feature_name}_{freq_band}": freq_val
                    for freq_band, freq_val in zip(freq_bands_means, weighted_means)
                }

                # Store in melspectrogram_dict
                for freq_band_name, freq_band_val in freq_bands_data.items():
                    
                    freq_band_stats = {f"{feature_name}_{freq_band_name}_{stats}": value for stats, value in zip(summary_statistics, get_summary_statistics(freq_band_val))}

                    for k, v in freq_band_stats.items():

                        if k not in melspectrogram_dict.keys():
                            melspectrogram_dict[k] = [v]
                        else:
                            melspectrogram_dict[k].append(v)

            # ---------------------------
            # C) MFCC, DELTA, DELTA2 (20, n)
            # Only first 13 rows used
            # ---------------------------
            elif feature_data.shape[0] == 20:
                reduced_feature_data = feature_data[:13, :]
                for i, row in enumerate(reduced_feature_data):
                    stats_values = get_summary_statistics(row.flatten())
                    # e.g. {"mfcc_0_mean": val, "mfcc_0_std": val, ...} for row i=0
                    statistical_summary = {
                        f"{feature_name}_{i}_{stat}": val
                        for stat, val in zip(summary_statistics, stats_values)
                    }
                    # Store in mfcc_deltas_dict
                    for key, val in statistical_summary.items():
                        if key not in mfcc_deltas_dict:
                            mfcc_deltas_dict[key] = [val]
                        else:
                            mfcc_deltas_dict[key].append(val)

            # ---------------------------
            # D) SPECTRAL CONTRAST (7, n)
            # ---------------------------
            elif feature_data.shape[0] == 7:
                for i, row in enumerate(feature_data):
                    stats_values = get_summary_statistics(row.flatten())
                    # e.g. {"spectral_contrast_0_mean": val, ...} for row i=0
                    statistical_summary = {
                        f"{feature_name}_{i}_{stat}": val
                        for stat, val in zip(summary_statistics, stats_values)
                    }
                    # Store in spectral_contrast_dict
                    for key, val in statistical_summary.items():
                        if key not in spectral_contrast_dict:
                            spectral_contrast_dict[key] = [val]
                        else:
                            spectral_contrast_dict[key].append(val)

    return results



import os
from hdf5_utils import save_dict_to_hdf5
from audio_preprocessing import process_audio
import librosa
    
def get_audio_features(file_path, lowcut, highcut, top_db):
    
    y, sr = librosa.load(file_path, sr=None)
    y_processed = process_audio(y=y, sr=sr, lowcut=lowcut, highcut=highcut, top_db=top_db)
    
    D = librosa.stft(y_processed)
    magnitude, phase = librosa.magphase(D)
    stft_power = magnitude**2
    
    melspectogram = librosa.feature.melspectrogram(S=stft_power, sr=sr)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(melspectogram), sr=sr)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    rms = librosa.feature.rms(S=magnitude)
    spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(S=magnitude, sr=sr)
    spectral_flatness = librosa.feature.spectral_flatness(S=magnitude)
    spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y_processed)
    f0 = librosa.yin(y=y_processed, sr=sr, fmin=60, fmax=700)

    features = {"melspectrogram": melspectogram,
                "mfcc": mfcc, 
                "delta_mfcc": delta_mfcc,
                "delta2_mfcc": delta2_mfcc,
                "spectral_contrast": spectral_contrast,
                "rms": rms,
                "spectral_centroid": spectral_centroid,
                "spectral_bandwidth": spectral_bandwidth,
                "spectral_flatness": spectral_flatness,
                "spectral_rolloff": spectral_rolloff,
                "zero_crossing_rate": zero_crossing_rate,
                "f0": f0}
    
    return features

def extract_audio_features(folder_path, lowcut, highcut, top_db):

    file_names = os.listdir(folder_path)
    file_names.sort(key= lambda x: int(x.split(".")[0]))
    
    total_files = len(file_names)  

    results = {}

    for i, file_name in enumerate(file_names):
        
        file_path = f"{folder_path}/{file_name}"
        
        print(f"Processing file {i + 1}/{total_files}: {file_name}")
        
        features = get_audio_features(file_path=file_path, lowcut=lowcut, highcut=highcut, topdb=top_db)
        results[file_name] = features

    location_path = f"main/data/audio_features_{folder_path.split('_')[-1]}.h5"
    save_dict_to_hdf5(dictionary=results, file_path=location_path)
    print(f"Feature extraction complete. Results saved to {location_path}")

if __name__ == "__main__":
    extract_audio_features(folder_path="main/data/audios_evaluation", lowcut=40, highcut=11025, top_db=35)
    extract_audio_features(folder_path="main/data/audios_development", lowcut=40, highcut=11025, top_db=35)
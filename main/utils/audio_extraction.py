import os
from hdf5_utils import save_dict_to_hdf5
from audio_preprocessing import process_audio
import librosa
    
def get_audio_features(file_path):

    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)

    # Apply noise removal, band-pass filtering and silence removal
    y_processed = process_audio(y=y, sr=sr)
    
    # Precompute reusable components
    D = librosa.stft(y_processed)
    magnitude, phase = librosa.magphase(D)
    stft_power = magnitude**2
    
    # Compute relevant features
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
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)

    features = {
        "melspectrogram": melspectogram,
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
        
    }
    
    # Return the features
    return features

def extract_audio_features(folder_path):

    file_names = os.listdir(folder_path)
    file_names.sort(key= lambda x: int(x.split(".")[0]))
    
    total_files = len(file_names)  

    results = {}

    for i, file_name in enumerate(file_names):
        
        file_path = f"{folder_path}/{file_name}"
        
        # Display progress
        print(f"Processing file {i + 1}/{total_files}: {file_name}")
        
        features = get_audio_features(file_path=file_path)
        results[file_name] = features

    location_path = f"main/data/audio_features_{folder_path.split('_')[-1]}.h5"
    save_dict_to_hdf5(dictionary=results, file_path=location_path)
    print(f"Feature extraction complete. Results saved to {location_path}")

if __name__ == "__main__":
    extract_audio_features(folder_path="main/data/audios_evaluation")
    extract_audio_features(folder_path="main/data/audios_development")
    

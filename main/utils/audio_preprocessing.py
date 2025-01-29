import numpy as np
import librosa
import noisereduce as nr

def reduce_noise(y, sr):
    y = nr.reduce_noise(y=y, sr=sr, stationary=False)
    return y

def bandpass_filter(y, sr, lowcut=45, highcut=11025):    
    D = librosa.stft(y=y)
    magnitude, phase = librosa.magphase(D=D)
    frequencies = librosa.fft_frequencies(sr=sr)
    
    mask = (frequencies >= lowcut) & (frequencies <= highcut)
    magnitude_filtered = magnitude * mask[:, None]

    D_filtered = magnitude_filtered * np.exp(1j * phase)
    y_filter = librosa.istft(D_filtered)
    return y_filter

def cut_silences(y, sr, top_db=45):

    interval_db = top_db + 10 
    
    y_trimmed, _ = librosa.effects.trim(y=y, top_db=top_db, ref=np.max)

    intervals = librosa.effects.split(y, top_db=interval_db)
    
    if intervals.shape[0] == 0:
        return y_trimmed
    
    y_no_pauses = y[intervals[0, 0]:intervals[0, 1]]

    for interval in intervals[1:]:
        segment = y[interval[0]:interval[1]]
        y_no_pauses = np.concatenate((y_no_pauses, segment))

    return y_no_pauses

def process_audio(y, sr, lowcut=40, highcut=11025, top_db=33):
    # Reduce noise
    try:
        y_noise = reduce_noise(y=y, sr=sr)
    except:
        print("Failed: {reduce_noise(y=y_emphasized, sr=sr)}")
        y_noise = y
    
    # Apply band-pass filter
    try:
        y_filter = bandpass_filter(y=y_noise, sr=sr, lowcut=lowcut, highcut=highcut)
    except:
        print("Failed: {bandpass_filter(y=y_noise, sr=sr, lowcut=lowcut, highcut=highcut)}")
        y_filter = y_noise

    # Remove leading, trailing and internal silences
    try:
        y_silence = cut_silences(y=y_filter, sr=sr, top_db=top_db)
    except:
        print("Failed: {cut_silences_dynamic(y=y_filter, sr=sr, percentile_parameter=percentile_parameter, perc_extra_db_parameter=perc_extra_db_parameter)}")
        y_silence = y_filter
    
    return y_silence

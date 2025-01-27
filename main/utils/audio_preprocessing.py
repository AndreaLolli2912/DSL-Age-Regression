import numpy as np
import librosa
import noisereduce as nr

def reduce_noise(y, sr):
    """
    Reduce background noise in the audio signal using the noisereduce library.

    PARAMETERS
    ----------
        y: 1D numpy array of audio samples.
        sr: Sampling rate (Hz).

    RETURNS
    -------
        y: Noise-reduced audio signal.
    """
    y = nr.reduce_noise(y=y, sr=sr, stationary=False)
    return y

def bandpass_filter(y, sr, lowcut=40, highcut=12000):
    """
    Apply a band-pass filter to retain frequencies within a specific range.
    
    PARAMETERS
    ----------
        y: 1D numpy array of audio samples.
        sr: Sampling rate (Hz).
        lowcut: Lower cutoff frequency (Hz).
        highcut: Upper cutoff frequency (Hz).
    
    RETURNS
    -------
        y: Band-pass filtered audio signal.
    """
    # Perform Short-Time Fourier Transform (STFT) to get the frequency domain
    D = librosa.stft(y=y)
    magnitude, phase = librosa.magphase(D=D)
    
    # Get the frequency values for STFT bins
    frequencies = librosa.fft_frequencies(sr=sr)
    
    # Create a mask for the desired frequency range (band-pass filter)
    mask = (frequencies >= lowcut) & (frequencies <= highcut)
    magnitude_filtered = magnitude * mask[:, None]
    
    # Reconstruct the signal using filtered magnitudes and original phases
    D_filtered = magnitude_filtered * np.exp(1j * phase)
    y_filter = librosa.istft(D_filtered)
    return y_filter

def cut_silences(y, sr, top_db=60):
    """
    Remove leading, trailing, and internal silences from an audio signal using a top_db threshold.

    PARAMETERS
    ----------
        y: 1D numpy array of audio samples.
        sr: Sampling rate (Hz).

    RETURNS
    -------
        y_no_pauses: Audio signal with silences removed.
    """
    # Dynamically calculate top_db based on the provided function
    top_db = top_db
    interval_db = top_db + 10 
    
    # Trim leading and trailing silence based on the dynamic decibel threshold
    y_trimmed, _ = librosa.effects.trim(y=y, top_db=top_db, ref=np.max)

    # Detect non-silent intervals in the trimmed audio
    intervals = librosa.effects.split(y, top_db=interval_db)
    
    if intervals.shape[0] == 0:
        return y_trimmed
    
    # Initialize the output signal with the first non-silent segment
    y_no_pauses = y[intervals[0, 0]:intervals[0, 1]]

    # Concatenate subsequent non-silent segments
    for interval in intervals[1:]:
        segment = y[interval[0]:interval[1]]
        y_no_pauses = np.concatenate((y_no_pauses, segment))

    return y_no_pauses

def process_audio(y, sr, lowcut=40, highcut=12000, top_db=60):
    """
    Process audio by applying noise reduction, band-pass filtering and silence removal.

    PARAMETERS
    ----------
        y: 1D numpy array of audio samples.
        sr: Sampling rate (Hz).
        lowcut: Lower cutoff frequency for band-pass filter (Hz).
        highcut: Upper cutoff frequency for band-pass filter (Hz)
        top_db: Threshold in decibels for silence detection

    RETURNS
    -------
        y: Processed audio signal after noise reduction , filtering and silence removal.
    """
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

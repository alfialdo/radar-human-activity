from scipy.signal import butter, filtfilt, stft
import numpy as np

class RadarPreproc():
    SAMPLE_PER_SWEEP = 128
    SWEEP = 0.001 # 1ms
    FS = SAMPLE_PER_SWEEP/SWEEP # Sampling rate 128 kHz
    FC = 5.8e9

    # def __init__(self, sample_per_sweep=128, chirp_duration=0.001, ):
    #     self.sps = sample_per_sweep
    #     self.chirp = chirp_duration
    #     self.fs = sample_per_sweep/chirp_duration

    @classmethod
    def cal_velocity(cls, freq):
        light_speed = 3e8
        wavelength = light_speed / cls.FC

        velocity = (freq * wavelength) / 2

        return velocity


    @classmethod
    def generate_pulses(cls, signal):
        num_sweeps = len(signal) // cls.SAMPLE_PER_SWEEP # This is number of chirps depends on the data N = 5000 or N = 10000
        pulses = signal.reshape(num_sweeps, cls.SAMPLE_PER_SWEEP)

        return pulses

    
    @classmethod
    def butterworth_highpass_filter(cls, fft_pulses, cutoff=0.0075, order=4, fs=False):
        # nyquist_freq = cls.FS * 0.5
        # normalize_cutoff = cutoff / nyquist_freq
        fs = fs if fs else None

        b, a = butter(N=order, Wn=cutoff, btype='highpass', fs=fs, analog=False)

        filtered_fft_p = filtfilt(b, a, fft_pulses, axis=0)

        return filtered_fft_p
        

    @classmethod
    def hamming_windowed_fft(cls, pulse):
        
        # Apply Hamming Window
        window = np.hamming(cls.SAMPLE_PER_SWEEP)
        windowed_pulse = pulse * window

        # Apply Fast Fourier Transform
        # Create range time map that represent power for Frequency from FFT results
        fft_p = np.fft.fft(windowed_pulse)

        # Create frequency axis for visualization
        fft_freq = np.fft.fftfreq(fft_p.size, d=1/cls.FS)

        return fft_p, fft_freq

    @classmethod
    def stft_doppler_signature(cls, filtered_sig, target_signature=203):
        # Set window and overlap
        window_length = int(0.2 * cls.FS)
        overlap = int(0.95 * window_length)

        # Convert filtered signal to magnitude (dB)
        mag = filtered_sig.ravel()

        f, t, Zxx = stft(mag, fs=cls.FS, return_onesided=False, window='hamming', nperseg=window_length, noverlap=overlap)

        # Apply fft shifting
        Zxx = np.fft.fftshift(Zxx, axes=0)
        f = np.fft.fftshift(f)

        # Filter target freq and magnitude using target_signature in Hz
        filter = np.where((f >= -target_signature) & (f <= target_signature))

        f = f[filter]
        Zxx = Zxx[filter]

        # Calculate velocity
        v = cls.cal_velocity(f)

        return f, t, Zxx, v


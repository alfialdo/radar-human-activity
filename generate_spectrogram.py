import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import os

from utils import Plot, RadarPreproc
from utils.common import load_sample, load_signal

from scipy.signal import stft, filtfilt
from multiprocessing import Pool
from argparse import ArgumentParser

ap = ArgumentParser()
ap.add_argument(
    '--dataset-path', type=str, default='../dataset',
    help='Path to Radar .dat dataset'
)

args = ap.parse_args()

def preprocess_data(sig):
    pulses = RadarPreproc.generate_pulses(sig)

    fft_pulses = []
    fft_freq = []

    for p in pulses:
        fft_p, fft_f = RadarPreproc.hamming_windowed_fft(p)
        fft_pulses.append(fft_p)
        fft_freq.append(fft_f)

    fft_pulses = np.asarray(fft_pulses)
    fft_freq = np.asarray(fft_freq)

    # Check shape of fft_pulses and fft_freq
    assert fft_pulses.shape == fft_freq.shape, 'fft_pulses and fft_freq shape are not same'

    fft_pulses_filetered = RadarPreproc.butterworth_highpass_filter(fft_pulses)

    f, t, Zxx, v = RadarPreproc.stft_doppler_signature(fft_pulses_filetered)

    return f, t, Zxx, v


def generate_spectrogram(data, label):
    label_path = f'data/{label}'

    if not os.path.exists(label_path):
        os.mkdir(label_path)

    for row in tqdm(data.itertuples(), total=len(data), desc=f'Generating spectrogram label {label}', position=int(label)):
        sig = load_signal(row.location, row.file, dataset_path=args.dataset_path)

        # IF walking then divide the data to 5s each
        if label == '0':
            cut = len(sig)//2
            sigs = [sig[:cut], sig[cut:]]

            for i, s in enumerate(sigs, 1):
                _, t, Zxx, v = preprocess_data(s)

                Plot.spectrogram(
                    v, t, Zxx, img=True,
                    label=label, file=f"{row.file}-{i}"
                )

        else:
            _, t, Zxx, v = preprocess_data(sig)

            Plot.spectrogram(
                v, t, Zxx, img=True,
                label=label, file=row.file
            )


if __name__ == '__main__':

    with open('data/label.json', 'r') as f:
        labels = json.load(f)

    df = pd.read_pickle('data/radar_dataset_cleaned.pickle')

    targets = [df.loc[df.label == x] for x in labels.values()]

    with Pool(processes=10) as pool:
        results = pool.starmap(generate_spectrogram, zip(targets, list(labels.keys())))

    
import torch
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from PIL import Image
import pandas as pd
import numpy as np
import json
import os

from utils import RadarPreproc

class RawRadarDataset():
    def __init__(self, data:pd.DataFrame):
        self.data = data

    def __len__(self):
        return len(self.data)


    # This should process our data into an tensor, apply transform function if any
    def __getitem__(self, idx):
        signal = self.__load_signal(self.data.loc[idx,'dat_file'])
        signal = self.__preprocess_data(signal)
        label = torch.tensor(self.data.loc[idx, 'label'])
        signal = torch.from_numpy(np.abs(signal)).float()


        return dict(signal=signal, label=label)
    
    @classmethod
    def create_train_test_data(cls, root_dir, df, seed=11, max_sample=310) -> pd.DataFrame:
        temp_df = dict(label=[], dat_file=[])
        labels = {"walking":0, "sitting":1, "standing":2, "picking_up_item":3, "drinking":4, "falling":5}

        for row in df.itertuples():
            temp_df['dat_file'].append(os.path.join(root_dir, row.location, f'{row.file}.dat'))
            temp_df['label'].append(labels[row.label])

        # Criteria for dataset splitting
        # max 310 images per class
        df = pd.DataFrame(temp_df)
        df = df.groupby('label').apply(lambda x: x.sample(min(len(x), max_sample), random_state=seed)).reset_index(drop=True)
        
        # Split train: 80%, test: 20%
        train_df = []
        test_df = []

        for label in df.label.unique():
            train, test = train_test_split(
                df.loc[df.label == int(label)],
                test_size=0.2,
                random_state = seed
            )

            train_df.append(train)
            test_df.append(test)

        train_df = pd.concat(train_df, ignore_index=True)
        test_df = pd.concat(test_df, ignore_index=True)

        
        return train_df, test_df

    def __load_signal(self, dat_file):
        with open(dat_file, 'r') as f:
            lines = f.readlines()
            sig = np.asanyarray([complex(x.replace('i', 'j')) for x in lines[4:]])

        if len(sig) > 640000:
            sig = sig[:640000]

        return sig
    
    def __preprocess_data(self, sig):
        # TODO: change to path
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

        # f, t, Zxx, v = RadarPreproc.stft_doppler_signature(fft_pulses_filetered)

        return fft_pulses_filetered
        


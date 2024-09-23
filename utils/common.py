import pandas as pd
import numpy as np

def load_sample(samples, dataset_path='../dataset'):

    data = dict(
        label = [], # .dat file name
        # fc = [], # carrier frequency
        file = [],
        # samples = [], # per recorded beat-note signal
        # bandwith = [], # Hz
        signal = [] # Complex value output from Radar data
    )
    # cols = list(data.keys())
    
    for s in samples.itertuples():

        dat = f'{dataset_path}/{s.location}/{s.file}.dat'

        data['label'].append(s.label)
        data['file'].append(s.file)

        with open(dat, 'r') as f:
            lines = f.readlines()
            
            data['signal'].append([complex(x.replace('i', 'j')) for x in lines[4:]])
    
    return pd.DataFrame(data)


def load_signal(location, file, dataset_path='../dataset'):
    dat = f'{dataset_path}/{location}/{file}.dat'

    with open(dat, 'r') as f:
        lines = f.readlines()    
        sig = np.asanyarray([complex(x.replace('i', 'j')) for x in lines[4:]])
    
    return sig

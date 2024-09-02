import os
import glob
import pandas as pd

dataset_path = "../dataset"
dirs = os.listdir(dataset_path)


for dir in dirs:
    if dir in ['dataframe','6February2019NGHomesDataset', '5February2019UoGDataset']:
        continue

    file_paths = sorted(glob.glob(f'{dataset_path}/{dir}/' + '*.dat'))
    print(f'Processing: {dir}...')
    
    data = dict(
        label = [], # .dat file name
        fc = [], # carrier frequency
        duration = [],
        samples = [], # per recorded beat-note signal
        bandwith = [], # Hz
        complexes = [] # Complex value output from Radar data
    )
    cols = list(data.keys())

    for dat in file_paths:
        try:
            data['label'].append(dat.split('.')[0])

            with open(dat, 'r') as f:
                lines = f.readlines()

                for i, line in enumerate(lines, 1):
                    if i < 5:
                        data[cols[i]].append(float(line))
                    else:
                        data[cols[-1]].append([complex(x.replace('i', 'j')) for x in lines[4:]])
                        break
        except Exception as e:
            print(dat, e)
            continue

    df = pd.DataFrame(data)
    print(f'Output: {dataset_path}/dataframe/{dir}.pickle')
    df.to_pickle(f'{dataset_path}/dataframe/{dir}.pickle')
    del df

    # break

        
import os
import glob
import pandas as pd
from argparse import ArgumentParser

### LOAD Data Description
ap = ArgumentParser()
ap.add_argument(
    '--dataset-path', type=str, default='../dataset',
    help='Path to Radar .dat dataset'
)

data = dict(
    file = [],
    location = [], 
    person_id = [],
    activity = [], 
    reps = [],
    complex_cnt = [] # Complex value output from Radar data
)


if __name__ == '__main__':
    args = ap.parse_args()

    dirs = os.listdir(args.dataset_path)

    for dir in dirs:

        file_paths = sorted(glob.glob(f'{args.dataset_path}/{dir}/' + '*.dat'))
        print(f'Processing: {dir}...')
        

        for dat in file_paths:
            try:
                label = dat.split('/')[-1].split('.')[0]
                data['file'].append(label)
                data['location'].append(dir)
                data['person_id'].append(label[1:4])
                data['activity'].append(label[4:7])
                data['reps'].append(label[7:])

                with open(dat, 'r') as f:
                    lines = f.readlines()

                    data['complex_cnt'].append(len(lines[4:]))

            except Exception as e:
                print(dat, e)
                continue

    raw_df = pd.DataFrame(data)

    ### General data cleaning
    # Clean file name
    raw_df = raw_df.loc[raw_df.file.str.len() <= 10]

    # Clean complexes count sample
    raw_df = raw_df.loc[raw_df.complex_cnt <= 1280000]

    # Clean wrong activity code
    raw_df = raw_df.loc[raw_df.activity.isin(['A01', 'A02', 'A03', 'A04', 'A05', 'A06'])]

    # Clean 0 in reps
    raw_df['reps'] = raw_df.reps.apply(lambda x: x.replace('0', ''))

    # Add duation
    raw_df['duration'] = raw_df.complex_cnt / (128 * 1000)

    # Clean outlier A02
    raw_df = raw_df.loc[~((raw_df.activity == 'A02') & (raw_df.duration > 5))]

    # Add labels
    labels = dict(A01='walking', A02='sitting', A03='standing', A04='picking_up_item', A05='drinking', A06='falling')
    raw_df['label'] = raw_df.activity.map(labels)

    print('Writing dataframe to radar_dataset_cleaned.pickle...')
    raw_df.to_pickle('data/radar_dataset_cleaned.pickle')
        
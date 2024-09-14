import torch
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from PIL import Image
import pandas as pd
import json
import os

class RadarDataset():
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)


    # This should process our data into an tensor, apply transform function if any
    def __getitem__(self, idx):
        img = Image.open(self.data.loc[idx,'img_path'])
        label = torch.tensor(self.data.loc[idx, 'label'])

        if self.transform:
            img = self.transform(img)

        return dict(image=img, label=label)
    
    @classmethod
    def create_train_test_data(cls, img_path, label_path, seed=11, max_sample=310) -> pd.DataFrame:
        with open(label_path, 'r') as f:
            labels = json.load(f)
        
        df = dict(label=[], img_path=[])

        for i in labels.keys():
            path = os.path.join(img_path, str(i))

            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)

                if os.path.isfile(file_path):
                    df['label'].append(int(i))
                    df['img_path'].append(file_path)

        # Criteria for dataset splitting
        # max 310 images per class
        df = pd.DataFrame(df)
        df = df.groupby('label').apply(lambda x: x.sample(min(len(x), max_sample), random_state=seed)).reset_index(drop=True)
        
        # Split train: 80%, test: 20%
        train_df = []
        test_df = []

        for label in labels.keys():
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
        
    # @classmethod
    # def plot_sample(cls, **radar):
    #     for i in len(radar['image']):
    #         plt.imshow(img)
    #         plt.show()


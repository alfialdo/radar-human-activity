import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from tqdm.auto import tqdm

from RadarClassifier.RadarDataset import RadarDataset
from RadarClassifier.ConvClassifier import ConvClassifier

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


def load_model(model_path, num_classes=6):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ConvClassifier(in_channels=3, image_size=(320,320), num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, device


def get_test_dataloader(batch_size=16, size=(320,320)):
    tsfm = transforms.Compose([
        transforms.Resize(size=size),
        transforms.ToTensor()
    ])

    _, test_df = RadarDataset.create_train_test_data(
        img_path='data',
        label_path='data/label.json'
    )

    radar_data_test = RadarDataset(test_df, transform=tsfm)
    test_loader = DataLoader(radar_data_test, batch_size=batch_size, shuffle=True, num_workers=16)

    return test_loader


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_images = []

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing model"):
            images, labels = data['image'].to(device), data['label'].to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_images.extend(images.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_images)
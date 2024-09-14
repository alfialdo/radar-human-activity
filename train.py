from RadarClassifier import RadarDataset
from RadarClassifier import ConvClassifier

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import transforms

from argparse import ArgumentParser
from tqdm import tqdm

import torch


ap = ArgumentParser()

ap.add_argument(
    '--epochs', type=int, default=10,
    help='Number of epoch for model training'
)

ap.add_argument(
    '--best-model-path', type=str, default='pretrained/best_model.pth',
    help='Path to save best model state during training'
)


def get_dataloader(batch_size):
    tsfm=transforms.Compose([
        transforms.Resize(size=(320,320)),
        transforms.ToTensor()
    ])

    train_df, test_df = RadarDataset.create_train_test_data(
        img_path='data',
        label_path='data/label.json'
    )

    radar_data_train = RadarDataset(train_df, transform=tsfm)
    radar_data_test = RadarDataset(test_df, transform=tsfm)

    train_loader = DataLoader(radar_data_train, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(radar_data_test, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

def cal_accuracy(true_pred, total_pred):
    return (true_pred / total_pred) * 100

def evaluate(test_loader, model, loss_func, device):
    model.eval()
    true_pred, total_pred = 0, 0
    val_loss = 0.0


    with torch.no_grad():
        for data in tqdm(test_loader, desc=f"Model validation"):
            image, label = data['image'].to(device), data['label'].to(device)

            outputs = model(image)
            loss = loss_func(outputs, label)

            val_loss += loss.item()

            _, pred = torch.max(outputs, 1)
            total_pred += label.size(0)
            true_pred += (pred == label).sum().item()

    return val_loss/len(test_loader), true_pred, total_pred

    

def train(model, optimizer, loss_func, epochs, device, model_path):
    metrics=dict(train_loss=[], val_loss=[])
    train_loader, test_loader = get_dataloader(batch_size=16)
    best_val_loss = 1000000.0
    model.to(device)

    for epoch in range(epochs):
        
        model.train()
        train_loss = 0.0
        
        for data in tqdm(train_loader, desc=f'Training epoch {epoch+1}/{epochs}'):
            image, label = data['image'].to(device), data['label'].to(device)

            optimizer.zero_grad()
            outputs = model(image)

            loss = loss_func(outputs, label)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        # Validate trained model
        val_loss, true_pred, total_pred = evaluate(test_loader, model, loss_func, device)
        metrics['train_loss'].append(train_loss/len(train_loader))
        metrics['val_loss'].append(val_loss)
        print(f"Epoch: {epoch+1}/{epochs} | train_loss: {train_loss/len(train_loader):.4f} | val_loss: {val_loss:.4f} | val_accuracy: {cal_accuracy(true_pred, total_pred):.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f'Save best model on epoch {epoch+1} to {model_path}...')
        
        


if __name__ == '__main__':
    args = ap.parse_args()
    conv_classifier = ConvClassifier(channels=3, image_size=(320,320), num_classes=6)

    train(
        model=conv_classifier,
        optimizer=optim.Adam(conv_classifier.parameters(), lr=0.001),
        loss_func=nn.CrossEntropyLoss(),
        epochs=args.epochs,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        model_path=args.best_model_path
    )

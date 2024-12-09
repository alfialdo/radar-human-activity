from RadarClassifier import RadarDataset
from RadarClassifier.ConvClassifier import ConvClassifier

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import transforms

from argparse import ArgumentParser
from tqdm import tqdm

import torch
from typing import NamedTuple

from RadarClassifier.TransformerClassifier import ViTClassifierModel

ap = ArgumentParser()

ap.add_argument(
    '--epochs', type=int, default=10,
    help='Number of epoch for model training'
)

ap.add_argument(
    '--best-model-path', type=str, default='pretrained/best_model.pth',
    help='Path to save best model state during training'
)

ap.add_argument(
    '--model', type=str, default='conv-block',
    help='Model architecture to use'
)

def get_dataloader(batch_size, img_size, workers):
    tsfm=transforms.Compose([
        transforms.Resize(size=(img_size[0],img_size[1])),
        transforms.ToTensor()
    ])

    train_df, test_df = RadarDataset.create_train_test_data(
        img_path='data',
        label_path='data/label.json'
    )

    radar_data_train = RadarDataset(train_df, transform=tsfm)
    radar_data_test = RadarDataset(test_df, transform=tsfm)

    train_loader = DataLoader(radar_data_train, batch_size=batch_size, shuffle=True, num_workers=workers)
    test_loader = DataLoader(radar_data_test, batch_size=batch_size, shuffle=False, num_workers=workers)

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

    

def train(model, optimizer, loss_func, epochs, device, model_path, batch_size, img_size, workers):
    metrics=dict(train_loss=[], val_loss=[])
    train_loader, test_loader = get_dataloader(batch_size=batch_size, img_size=img_size, workers=workers)
    best_val_acc = 0.0
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
        val_acc = cal_accuracy(true_pred, total_pred)
        print(f"Epoch: {epoch+1}/{epochs} | train_loss: {train_loss/len(train_loader):.4f} | val_loss: {val_loss:.4f} | val_accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f'Save best model on epoch {epoch+1} to {model_path}...')
        
        


if __name__ == '__main__':
    args = ap.parse_args()
    class Config(NamedTuple):
        # Data Loader
        workers = 16
        batch_size = 32
        img_size = (80,80)
        
        # Training config
        epochs = 60
        lr = 0.001
        num_classes = 6
        device = torch.device('cuda')

        # Model hyperparameters
        patch_size = 8
        num_patches = (img_size[0] // patch_size) ** 2
        projection_dim = 32
        num_heads = 4
        transformer_layers = 8
        mlp_head_units = [2048, 1024]
        feed_forward_dim = projection_dim * 2
        seed = 11
        

    config = Config()

    if args.model == 'conv-block':
        model = ConvClassifier(in_channels=3, image_size=config.img_size, num_classes=config.num_classes)
    elif args.model == 'transformer':
        model = ViTClassifierModel(config, num_classes=config.num_classes, device=config.device)

    model = model.to(config.device)

    train(
        model=model,
        optimizer=optim.Adam(model.parameters(), lr=config.lr),
        loss_func=nn.CrossEntropyLoss(),
        epochs=config.epochs,
        device=config.device,
        model_path=args.best_model_path,
        batch_size=config.batch_size,
        img_size=config.img_size,
        workers=config.workers
    )

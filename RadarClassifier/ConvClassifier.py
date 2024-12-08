from torch import nn
from torch.nn import functional as F
from torch import flatten

class ConvClassifier(nn.Module):
    def __init__(self, num_classes, in_channels, image_size):
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.image_size = image_size

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = None

    def __init_classifier(self, x):
        # Forward pass through features
        x = self.backbone(x)

        # Get flattened dimension
        feat_dim = x.shape[1] * x.shape[2] * x.shape[3]
        
        # Initialize classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes)
        )

    def forward(self, x):
        if self.classifier is None:
            self.__init_classifier(x)
            self.classifier.to(x.device)

        x = self.backbone(x)
        x = self.classifier(x)

        # Output
        return x

    

        


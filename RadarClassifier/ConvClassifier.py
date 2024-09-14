from torch import nn
from torch.nn import functional as F
from torch import flatten

class ConvClassifier(nn.Module):
    def __init__(self, num_classes, channels, image_size):
        super().__init__()

        self.num_classes = num_classes
        self.channels = channels
        self.image_size = image_size

        self.conv1 = nn.Conv2d(self.channels, 64, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3,3))

        self.max_pool = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(128 * 78 * 78, 256)
        self.fc2 = nn.Linear(256, self.num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))

        # Output
        return self.fc2(x)

    

        


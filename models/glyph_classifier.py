import torch.nn as nn

from data.glyph_classes import GLYPH_CLASSES

MODEL_PATH = "checkpoints/glyph_classifier.pth"


# Define the enhanced CNN model for handling complex glyph features
class GlyphClassifier(nn.Module):
    def __init__(self):
        super(GlyphClassifier, self).__init__()

        # Convolutional Layers with increasing complexity
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, padding=1
        )  # Input channels = 3, output channels = 64
        self.conv2 = nn.Conv2d(
            64, 128, kernel_size=3, padding=1
        )  # Increase channels to 128
        self.conv3 = nn.Conv2d(
            128, 256, kernel_size=3, padding=1
        )  # Increase channels to 256
        self.conv4 = nn.Conv2d(
            256, 512, kernel_size=3, padding=1
        )  # Increase channels to 512
        self.conv5 = nn.Conv2d(
            512, 512, kernel_size=3, padding=1
        )  # Keep channels at 512 for further depth

        # Batch Normalization Layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(512)

        # Fully Connected Layers
        self.fc1 = nn.Linear(
            512 * 1 * 1, 1024
        )  # Adjust input size based on output of conv layers
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 127)
        self.fc5 = nn.Linear(127, len(GLYPH_CLASSES))

        # Pooling, Activation, and Dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Dropout with 50% probability

    def forward(self, x):
        # Convolutional layers with ReLU, BatchNorm, and Pooling
        x = self.pool(
            self.bn1(self.relu(self.conv1(x)))
        )  # Conv1 -> ReLU -> BatchNorm -> Pool
        x = self.pool(
            self.bn2(self.relu(self.conv2(x)))
        )  # Conv2 -> ReLU -> BatchNorm -> Pool
        x = self.pool(
            self.bn3(self.relu(self.conv3(x)))
        )  # Conv3 -> ReLU -> BatchNorm -> Pool
        x = self.pool(
            self.bn4(self.relu(self.conv4(x)))
        )  # Conv4 -> ReLU -> BatchNorm -> Pool
        x = self.pool(
            self.bn5(self.relu(self.conv5(x)))
        )  # Conv5 -> ReLU -> BatchNorm -> Pool

        # Flatten the output for the fully connected layers
        x = x.view(-1, 512 * 1 * 1)

        # Fully connected layers with ReLU and Dropout
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.fc5(x)  # Output layer

        return x

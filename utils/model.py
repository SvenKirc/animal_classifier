from torch import nn


def cnn_block(input_channels, output_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class Animal_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            cnn_block(3, 8),
            cnn_block(8, 16),
            cnn_block(16, 32),
            cnn_block(32, 64),
            cnn_block(64, 128),
            cnn_block(128, 256)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
from __future__ import annotations
import torch.nn as nn

"""The Covolutional Neural Network below is a small 7-layer NN, for usgin Cifar-10 dataset."""
class SmallCifarCnn(nn.Module):
    def __init__(self, NumClasses: int = 10):
        super().__init__()
        self.FeatureExtractor = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )
        self.Classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, NumClasses),
        )

    def forward(self, InputTensor):
        FeatureTensor = self.FeatureExtractor(InputTensor)
        return self.Classifier(FeatureTensor)


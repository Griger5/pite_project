import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        num_classes = 2

        def cnn_block(in_size, out_size):
            return nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=True),
            )

        self.net = nn.Sequential(
            cnn_block(1, 32),
            cnn_block(32, 64),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.10),
            cnn_block(64, 128),
            cnn_block(128, 256),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.15),
            cnn_block(256, 512),
            cnn_block(512, 512),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.20),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).unsqueeze(1)
        x = self.net(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channel, f, filters, s):
        super().__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=F1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(),
            nn.Conv2d(in_channels=F1, out_channels=F2, kernel_size=f, stride=s, padding=1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(),
            nn.Conv2d(in_channels=F2, out_channels=F3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3)
        )

        self.shortcut_1 = nn.Conv2d(in_channel, F3, kernel_size=f, stride=s, padding=1, bias=False)
        self.batch_1 = nn.BatchNorm2d(F3)
        self.relu_1 = nn.ReLU()

    def forward(self, X):
        X_shortcut = self.shortcut_1(X)
        X_shortcut = self.batch_1(X_shortcut)
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X

class IdentityBlock(nn.Module):
    def __init__(self, in_channel, f, filters):
        super().__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channels = in_channel, out_channels = F1, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(F1),
            nn.ReLU(),
            nn.Conv2d(in_channels = F1, out_channels = F2, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(F2),
            nn.ReLU(),
            nn.Conv2d(in_channels = F2, out_channels = F3, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(F3)
        )

        self.relu_1 = nn.ReLU()

    def forward(self, X):
        X_shortcut = X
        X = self.stage(X)
        X = X_shortcut + X
        X = self.relu_1(X)
        return X

class ResNet50(nn.Module):
    def __init__(self, n_class = 1000):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, padding = 1)
        )

        self.stage2 = nn.Sequential(
            ConvBlock(64, 3, [64, 256, 256], 1),
            IdentityBlock(256, 3, [64, 256, 256]),
            IdentityBlock(256, 3, [64, 256, 256]),
        )

        self.stage3 = nn.Sequential(
            ConvBlock(256, 3, [128, 128, 512], 2),
            IdentityBlock(512, 3, [128, 128, 512]),
            IdentityBlock(512, 3, [128, 128, 512]),
            IdentityBlock(512, 3, [128, 128, 512]),
            IdentityBlock(512, 3, [128, 128, 512]),
            IdentityBlock(512, 3, [128, 128, 512])
        )

        self.stage4 = nn.Sequential(
            ConvBlock(512, 3, [256, 256, 1024], 2),
            IdentityBlock(1024, 3, [256, 256, 1024]),
            IdentityBlock(1024, 3, [256, 256, 1024]),
            IdentityBlock(1024, 3, [256, 256, 1024]),
            IdentityBlock(1024, 3, [256, 256, 1024]),
            IdentityBlock(1024, 3, [256, 256, 1024])
        )

        self.stage5 = nn.Sequential(
            ConvBlock(1024, 3, [512, 512, 2048], 2),
            IdentityBlock(2048, 3, [512, 512, 2048]),
            IdentityBlock(2048, 3, [512, 512, 2048]),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, n_class)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


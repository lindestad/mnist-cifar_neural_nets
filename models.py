import torch.nn as nn
import torchvision.models as models


# ---------- MLPs ----------
class SmallMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class LargeMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------- simple CNN ----------
class SimpleCNN(nn.Module):
    """3-conv CNN for 32x32 RGB or 28x28 grayscale (channel-agnostic)."""

    def __init__(self, in_ch: int, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # /2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # /4
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # /8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4 if in_ch == 3 else 128 * 3 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ---------- ResNet-18 (adapted) ----------
def make_resnet18(in_ch: int, num_classes: int = 10):
    """
    Returns a fresh ResNet-18 modified for small inputs (32x32/28x28) and
    arbitrary channel count (1 or 3).
    """
    resnet = models.resnet18(weights=None)
    # adapt first conv (orig: 7×7, stride 2, padding 3)
    resnet.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False)
    resnet.maxpool = nn.Identity()  # remove 3×3 max-pool to keep resolution
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    return resnet

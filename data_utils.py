from pathlib import Path
from typing import Tuple, Literal
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image


# ---------- basic constants ----------
MNIST_SIZE = (1, 28, 28)
CIFAR_SIZE = (3, 32, 32)
NUM_CLASSES = 10


# ---------- helper dataset ----------
class OpenMLTensorDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        img_shape: Tuple[int, int, int],
        cnn_mode: bool,
        split: Literal["train", "val", "test"],
        augment: bool = False,
    ):
        super().__init__()
        c, h, w = img_shape
        self.X = X.astype(np.uint8).reshape(-1, h, w, c)  # NHWC
        self.y = y.astype(np.int64)
        self.cnn_mode = cnn_mode
        self.split = split

        # choose transforms -----------------
        base = []
        if augment:
            # only for training
            if c == 3:  # CIFAR-10
                base += [
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(15),
                    T.RandomCrop(w, padding=4),
                ]
            else:  # MNIST
                base += [
                    T.RandomRotation(15),
                ]
        base += [T.ToTensor()]  # converts to C×H×W & /255
        self.tf = T.Compose(base)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img_np = self.X[idx]
        img_pil = Image.fromarray(
            img_np.squeeze() if img_np.shape[-1] == 1 else img_np,
            mode="L" if img_np.shape[-1] == 1 else "RGB",
        )
        img = self.tf(img_pil)  # tensor in [0,1]
        if not self.cnn_mode:
            img = torch.flatten(img)  # for MLP
        label = torch.tensor(self.y[idx])
        return img, label


# ---------- public API ----------
def get_dataloaders(
    dataset: Literal["mnist", "cifar"],
    model_type: Literal["mlp", "cnn"],
    batch_size: int = 64,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader) for requested dataset/model type.
    """

    # ---------- download ----------
    if dataset == "mnist":
        print("Fetching MNIST … (first call may take a while)")
        raw = fetch_openml("mnist_784", version=1, as_frame=False, cache=True)
        X, y = raw["data"], raw["target"].astype(np.int64)
        img_shape = MNIST_SIZE
    else:
        print("Fetching CIFAR-10 …")
        raw = fetch_openml("CIFAR_10_SMALL", version=1, as_frame=False, cache=True)
        X, y = raw["data"], raw["target"].astype(np.int64)
        img_shape = CIFAR_SIZE

    # ---------- splits ----------
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.20, stratify=y_trainval, random_state=42
    )

    # ---------- datasets ----------
    cnn_mode = model_type == "cnn"
    ds_train = OpenMLTensorDataset(
        X_train,
        y_train,
        img_shape=img_shape,
        cnn_mode=cnn_mode,
        split="train",
        augment=True,
    )
    ds_val = OpenMLTensorDataset(
        X_val, y_val, img_shape=img_shape, cnn_mode=cnn_mode, split="val", augment=False
    )
    ds_test = OpenMLTensorDataset(
        X_test,
        y_test,
        img_shape=img_shape,
        cnn_mode=cnn_mode,
        split="test",
        augment=False,
    )

    # ---------- dataloaders ----------
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(ds_train, shuffle=True, **loader_args)
    val_loader = DataLoader(ds_val, shuffle=False, **loader_args)
    test_loader = DataLoader(ds_test, shuffle=False, **loader_args)
    return train_loader, val_loader, test_loader

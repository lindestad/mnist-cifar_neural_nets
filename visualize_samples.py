# Draw 5x5 grids of random MNIST & CIFAR-10 images.

# Run:
#   python visualize_samples.py mnist
#   python visualize_samples.py cifar


import argparse, random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from torchvision.utils import make_grid
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["mnist", "cifar"])
    args = parser.parse_args()

    if args.dataset == "mnist":
        raw = fetch_openml("mnist_784", version=1, as_frame=False, cache=True)
        X = raw["data"].astype(np.uint8).reshape(-1, 1, 28, 28)
        y = raw["target"].astype(int)
        cmap, nch = "gray", 1
    else:
        raw = fetch_openml("CIFAR_10_SMALL", version=1, as_frame=False, cache=True)
        X = raw["data"].astype(np.uint8).reshape(-1, 3, 32, 32)
        y = raw["target"].astype(int)
        cmap, nch = None, 3

    idxs = random.sample(range(len(X)), 25)
    imgs = torch.tensor(X[idxs]) / 255.0  # to [0,1] tensor
    grid = make_grid(imgs, nrow=5, pad_value=1.0)

    plt.figure(figsize=(5, 5))
    if nch == 1:
        plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap=cmap)
    else:
        plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    dsname = args.dataset.upper()
    plt.title(f"{dsname} sample images")
    out = Path("outputs") / args.dataset / "samples.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()

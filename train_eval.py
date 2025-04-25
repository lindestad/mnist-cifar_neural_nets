# Unified training + evaluation script for any dataset/model.

# Usage examples:
#   python train_eval.py --dataset mnist --model small_mlp
#   python train_eval.py --dataset cifar --model resnet
# Outputs: metrics & curves under outputs/<dataset>/<model>/...

import argparse, json, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_utils import get_dataloaders, MNIST_SIZE, CIFAR_SIZE
from models import SmallMLP, LargeMLP, SimpleCNN, make_resnet18


# ---------- helpers ----------
def save_curves(outdir: Path, history: dict):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid()
    plt.title("Cross-entropy loss")
    plt.savefig(outdir / "loss_curve.png", dpi=300)

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train")
    plt.plot(epochs, history["val_acc"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.grid()
    plt.title("Accuracy")
    plt.savefig(outdir / "acc_curve.png", dpi=300)
    plt.close("all")


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    crit = nn.CrossEntropyLoss()
    total_loss, correct, n = 0.0, 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = crit(logits, y)
            total_loss += loss.item() * X.size(0)
            _, preds = logits.max(1)
            correct += (preds == y).sum().item()
            n += X.size(0)
            y_true.append(y.cpu())
            y_pred.append(preds.cpu())
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    return total_loss / n, correct / n, y_true.numpy(), y_pred.numpy()


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    patience: int = 5,
    lr: float = 1e-3,
):
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best_state, best_val_loss, stagn = None, float("inf"), 0

    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, n = 0.0, 0, 0

        for X, y in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            X, y = X.to(device), y.to(device)
            optim.zero_grad()
            logits = model(X)
            loss = crit(logits, y)
            loss.backward()
            optim.step()

            running_loss += loss.item() * X.size(0)
            _, preds = logits.max(1)
            correct += (preds == y).sum().item()
            n += X.size(0)

        train_loss = running_loss / n
        train_acc = correct / n
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device)

        hist["train_loss"].append(train_loss)
        hist["train_acc"].append(train_acc)
        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(val_acc)

        print(
            f"[{epoch:02d}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.3f} "
            f"val_acc={val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            stagn = 0
        else:
            stagn += 1
            if stagn >= patience:
                print("Early-stopping.")
                break

    model.load_state_dict(best_state)
    return model, hist


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "cifar"], required=True)
    parser.add_argument(
        "--model",
        choices=["small_mlp", "large_mlp", "simple_cnn", "resnet"],
        required=True,
    )
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(f"outputs/{args.dataset}/{args.model}")
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------- data ----------
    model_type = "cnn" if "cnn" in args.model or args.model == "resnet" else "mlp"
    train_ld, val_ld, test_ld = get_dataloaders(
        dataset=args.dataset,
        model_type=model_type,
        batch_size=args.batch,
    )

    # ---------- model ----------
    if args.dataset == "mnist":
        input_dim = 28 * 28
        in_ch = 1
    else:
        input_dim = 32 * 32 * 3
        in_ch = 3

    if args.model == "small_mlp":
        model = SmallMLP(input_dim)
    elif args.model == "large_mlp":
        model = LargeMLP(input_dim)
    elif args.model == "simple_cnn":
        model = SimpleCNN(in_ch)
    else:  # resnet
        model = make_resnet18(in_ch)

    # ---------- train ----------
    t0 = time.perf_counter()
    model, hist = train_model(
        model, train_ld, val_ld, device, epochs=args.epochs, patience=args.patience
    )
    print(f"training finished in {(time.perf_counter()-t0)/60:.1f} minutes")

    # ---------- curves ----------
    save_curves(outdir, hist)

    # ---------- test ----------
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_ld, device)
    print(f"TEST  loss={test_loss:.4f}  acc={test_acc:.3f}")

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # save metrics ----------
    with open(outdir / "classification_report.json", "w") as fp:
        json.dump(report, fp, indent=4)
    np.save(outdir / "confusion_matrix.npy", cm)

    # pretty confusion-matrix plot ----------
    import seaborn as sns

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted")
    plt.savefig(outdir / "confusion_matrix.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()

# Generate simple gradient-based saliency maps.

# If the expected checkpoint  outputs/<dataset>/<model>/best.pt  is not found,
# the script automatically trains the model first (early-stopping, max 50 epochs)
# and then produces the saliency maps.

# Example
# -------
# python saliency.py --dataset mnist --model simple_cnn --n 6

from __future__ import annotations
import argparse, random, time
from pathlib import Path

import torch, torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_utils import get_dataloaders
from models import SimpleCNN, make_resnet18


# ---------------------- core helpers  ------------------------------------
def train_if_needed(
    model: nn.Module,
    ckpt: Path,
    dataset: str,
    device: torch.device,
    *,
    batch_size: int = 64,
    max_epochs: int = 50,
    patience: int = 5,
) -> None:
    """Train *only* when the checkpoint does not exist."""
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"[✓] Loaded weights from {ckpt}")
        return

    print(f"[!] {ckpt} not found – training a fresh model (this may take a while)")
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    # dataloaders ------------------------------------------------------------
    train_ld, val_ld, _ = get_dataloaders(
        dataset=dataset,
        model_type="cnn",
        batch_size=batch_size,
    )

    crit = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val, stagn = float("inf"), 0

    for epoch in range(1, max_epochs + 1):
        # ---------- train ---------------------------------------------------
        model.train()
        running_loss, correct, n = 0.0, 0, 0
        for X, y in tqdm(train_ld, desc=f"epoch {epoch}", leave=False):
            X, y = X.to(device), y.to(device)
            optim.zero_grad()
            logits = model(X)
            loss = crit(logits, y)
            loss.backward()
            optim.step()

            running_loss += loss.item() * X.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += X.size(0)

        train_loss = running_loss / n
        train_acc = correct / n

        # ---------- validation ---------------------------------------------
        model.eval()
        with torch.no_grad():
            v_loss, v_correct, n = 0.0, 0, 0
            for X, y in val_ld:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                loss = crit(logits, y)
                v_loss += loss.item() * X.size(0)
                v_correct += (logits.argmax(1) == y).sum().item()
                n += X.size(0)
            val_loss = v_loss / n
            val_acc = v_correct / n

        print(
            f"[{epoch:02d}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} train_acc={train_acc:.3f} "
            f"val_acc={val_acc:.3f}"
        )

        # ---------- early-stopping -----------------------------------------
        if val_loss < best_val:
            best_val, stagn = val_loss, 0
            torch.save(model.state_dict(), ckpt)
        else:
            stagn += 1
            if stagn >= patience:
                print("Early stopping.")
                break

    print(f"[✓] Training finished – best model saved to {ckpt}")


def compute_saliency(model: nn.Module, img: torch.Tensor, label: int, device):
    """
    Vanilla gradient saliency. Adds a batch dimension only if it isn't present.
    """
    model.eval()

    # ensure shape is (1, C, H, W)
    if img.dim() == 3:  # (C, H, W)
        img = img.unsqueeze(0)
    img = img.to(device).requires_grad_(True)

    out = model(img)
    out[0, label].backward()

    # remove batch, average over channels  -> (H, W)
    sal = img.grad.detach().abs().squeeze(0).mean(dim=0)
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    return sal.cpu()


# --------------------------------------------------------------------------- #
#                                   main                                      #
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["mnist", "cifar"], required=True)
    p.add_argument("--model", choices=["simple_cnn", "resnet"], required=True)
    p.add_argument("--n", type=int, default=5, help="# images to visualise")
    p.add_argument("--batch", type=int, default=64, help="batch size for retraining")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model -----------------------------------------------------------
    in_ch = 1 if args.dataset == "mnist" else 3
    model = SimpleCNN(in_ch) if args.model == "simple_cnn" else make_resnet18(in_ch)
    model.to(device)

    ckpt = Path(f"outputs/{args.dataset}/{args.model}/best.pt")
    train_if_needed(model, ckpt, args.dataset, device, batch_size=args.batch)

    # dataloader (test split) ----------------------------------------------
    _, _, test_ld = get_dataloaders(
        dataset=args.dataset,
        model_type="cnn",
        batch_size=1,
    )

    outdir = ckpt.parent / "saliency"
    outdir.mkdir(parents=True, exist_ok=True)

    # pick N random images --------------------------------------------------
    samples = random.sample(list(test_ld), args.n)
    for i, (img, label) in enumerate(samples, 1):
        sal = compute_saliency(model, img, label.item(), device)

        # --- prepare the original image for plotting ----------------------
        if img.dim() == 4:  # (1, C, H, W)
            vis = img.squeeze(0)  # -> (C, H, W)
        else:  # already (C, H, W) or (H, W)
            vis = img
        if vis.dim() == 3:  # (C, H, W)  ➜  (H, W, C)
            img_np = vis.permute(1, 2, 0).cpu().numpy()
        else:  # (H, W)
            img_np = vis.cpu().numpy()

        # --- plot ---------------------------------------------------------
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        if in_ch == 1:
            plt.imshow(img_np.squeeze(), cmap="gray")
        else:
            plt.imshow(img_np)
        plt.title(f"input  (class={label.item()})")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(sal, cmap="hot")
        plt.title("saliency")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(outdir / f"saliency_{i}.png", dpi=300)
        plt.close()

    print("Saliency maps saved ->", outdir)


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"Done in {(time.perf_counter() - t0):.1f}s")

"""
Fine-tuning ResNet-18 (pretrained ImageNet) untuk sendok vs garpu,
mengikuti alur notebook Colab:
  - split train/val/test = 80/10/10 (random_split, seed tetap)
  - Adam + CrossEntropyLoss, early stopping (patience 3) berdasarkan val loss
  - simpan best state, evaluasi di test set dengan classification_report
  - grafik: kurva loss, kurva akurasi, confusion matrix, latency

Struktur dataset:
    dataset/
        garpu/
        sendok/

Pemakaian:
    pip install torch torchvision numpy tqdm scikit-learn matplotlib seaborn
    python3 train_resnet18.py --data_dir dataset
"""

import argparse
import copy
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn, optim
from torch.utils import data
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="dataset")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--out", default="resnet18_sendok_garpu.pth")
    p.add_argument("--show", action="store_true", help="tampilkan grafik di jendela")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class ResNet18Model(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.resnet = models.resnet18(weights=weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        # fine-tuning penuh: semua layer ikut dilatih
        for p in self.resnet.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.resnet(x)


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
def run_epoch(dataloader, model, loss_fn, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    preds, trues, total_loss, n = [], [], 0.0, 0

    with torch.set_grad_enabled(train):
        for inputs, labels in tqdm(dataloader, leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            preds.append(outputs.argmax(1).cpu())
            trues.append(labels.cpu())
            total_loss += loss.item() * inputs.size(0)
            n += inputs.size(0)

    return total_loss / n, torch.cat(trues).numpy(), torch.cat(preds).numpy()


# --------------------------------------------------------------------------- #
# Latency
# --------------------------------------------------------------------------- #
@torch.no_grad()
def measure_latency(model, device, batch_size=1, img_size=224, warmup=20, runs=200):
    model = model.to(device).eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    cuda = device.type == "cuda"
    for _ in range(warmup):
        model(x)
    if cuda:
        torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        if cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        model(x)
        if cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1000)
    t = np.array(times)
    return {"times": t, "mean": t.mean(), "std": t.std(), "median": np.median(t),
            "p95": np.percentile(t, 95), "p99": np.percentile(t, 99),
            "fps": 1000.0 / t.mean() * batch_size}


def print_latency(name, r):
    print(f"[{name}] mean {r['mean']:.2f} ms (+/- {r['std']:.2f}) | median {r['median']:.2f} | "
          f"p95 {r['p95']:.2f} | p99 {r['p99']:.2f} | FPS {r['fps']:.1f}")


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def plot_curve(train, val, title, ylabel, fname):
    epochs = range(1, len(train) + 1)
    plt.figure(figsize=(6, 4))
    sns.lineplot(x=epochs, y=train, marker="o")
    sns.lineplot(x=epochs, y=val, marker="o")
    plt.legend(["train", "validation"])
    plt.xlabel("EPOCH")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)


def plot_confusion(y_true, y_pred, classes, fname):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)


def plot_latency(results, fname):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    first_name, first = next(iter(results.items()))
    ax[0].hist(first["times"], bins=30, color="steelblue")
    ax[0].axvline(first["mean"], color="red", linestyle="--", label=f"mean {first['mean']:.1f} ms")
    ax[0].set_title(f"Distribusi Latency ({first_name})")
    ax[0].set_xlabel("ms")
    ax[0].set_ylabel("jumlah")
    ax[0].legend()

    names = list(results.keys())
    means = [results[k]["mean"] for k in names]
    p95 = [results[k]["p95"] for k in names]
    x = np.arange(len(names))
    ax[1].bar(x - 0.2, means, 0.4, label="mean")
    ax[1].bar(x + 0.2, p95, 0.4, label="p95")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(names, rotation=15)
    ax[1].set_ylabel("ms")
    ax[1].set_title("Perbandingan Latency")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=150)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = get_args()
    random.seed(40)
    np.random.seed(40)
    torch.manual_seed(40)
    torch.cuda.manual_seed_all(40)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Data ----
    basic_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_dataset = ImageFolder(args.data_dir, basic_transforms)
    classes = img_dataset.classes

    generator = torch.Generator().manual_seed(42)
    train_set, valid_set = data.random_split(img_dataset, [0.8, 0.2], generator=generator)
    valid_set, test_set = data.random_split(valid_set, [0.5, 0.5], generator=generator)
    print(f"Kelas: {classes} | train {len(train_set)} | val {len(valid_set)} | test {len(test_set)}")

    train_dl = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_dl = data.DataLoader(valid_set, batch_size=args.batch_size)
    test_dl = data.DataLoader(test_set, batch_size=args.batch_size)

    # ---- Model ----
    model = ResNet18Model(num_classes=len(classes), pretrained=True).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ---- Training loop + early stopping ----
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_loss, best_state, patience = float("inf"), None, 0
    t0 = time.time()

    for i in range(args.epochs):
        print(f"\nTraining epoch {i + 1}/{args.epochs}")
        loss, yt, yp = run_epoch(train_dl, model, loss_fn, device, optimizer)
        acc = accuracy_score(yt, yp)
        history["train_loss"].append(loss)
        history["train_acc"].append(acc)
        print(f"train loss : {loss:.4f}, train acc : {acc:.4f}")

        loss, yt, yp = run_epoch(val_dl, model, loss_fn, device)
        acc = accuracy_score(yt, yp)
        history["val_loss"].append(loss)
        history["val_acc"].append(acc)
        print(f"val loss   : {loss:.4f}, val acc   : {acc:.4f}")

        if loss < best_loss:
            best_loss = loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"early stop at epoch {i + 1}")
                break

    print(f"\nTraining selesai dalam {(time.time() - t0) / 60:.1f} menit")

    # ---- Grafik training ----
    plot_curve(history["train_loss"], history["val_loss"], "Kurva Loss", "Loss", "kurva_loss.png")
    plot_curve(history["train_acc"], history["val_acc"], "Kurva Akurasi", "Akurasi", "kurva_akurasi.png")

    # ---- Evaluasi test set dengan model terbaik ----
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "classes": classes}, args.out)
    print(f"Model terbaik disimpan: {args.out}")

    loss, yt, yp = run_epoch(test_dl, model, loss_fn, device)
    acc = accuracy_score(yt, yp)
    print("\nResNet18 - Test Set")
    print(classification_report(yt, yp, target_names=classes, zero_division=0))
    print(f"test loss : {loss:.4f}, test acc : {acc:.4f}")
    plot_confusion(yt, yp, classes, "confusion_matrix.png")

    # ---- Latency ----
    print("\n=== PENGUKURAN LATENCY ===")
    results = {}
    results["CPU bs=1"] = measure_latency(copy.deepcopy(model).cpu(), torch.device("cpu"), 1, runs=100)
    print_latency("CPU bs=1", results["CPU bs=1"])
    if torch.cuda.is_available():
        for bs in (1, 8, 32):
            k = f"GPU bs={bs}"
            results[k] = measure_latency(copy.deepcopy(model), torch.device("cuda"), bs)
            print_latency(k, results[k])
    plot_latency(results, "latency.png")

    print("\nGrafik tersimpan: kurva_loss.png, kurva_akurasi.png, confusion_matrix.png, latency.png")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
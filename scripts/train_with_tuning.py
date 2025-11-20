import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from itertools import product
import json

proj_root = Path(__file__).parent.parent
sys.path.insert(0, str(proj_root))

from model.ocr_model_v2 import OCRModelV2
from data.word_dataset import WordOCRDataset, WordImageGenerator


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_charset(charset_path="charset.txt"):
    with open(charset_path, "r", encoding="utf-8") as f:
        return list(f.read().strip())


def prepare_word_dataset(cfg, overwrite=False):
    """Generate word images if they don't exist."""
    output_dir = cfg["output_dir"]
    existing = len([f for f in os.listdir(output_dir) if f.endswith(".png")]) if os.path.exists(output_dir) else 0

    if existing > 0 and not overwrite:
        print(f"✓ Found {existing} existing images in {output_dir}")
        return

    print("📊 Generating word dataset...")
    from datasets import load_dataset
    dataset = load_dataset("Sakonii/nepalitext-language-model-dataset")
    train_texts = dataset["train"]["text"]

    import re
    words = []
    for text in train_texts:
        if isinstance(text, str):
            matches = re.findall(r"[\u0900-\u097F]+", text)
            words.extend(matches)

    words = list(set(words))
    random.shuffle(words)
    words = words[: cfg["train_samples"]]

    print(f"✓ Using {len(words)} unique words")

    generator = WordImageGenerator(
        fonts_dir=cfg["fonts_dir"],
        output_dir=output_dir,
        font_size_range=(36, 56),
    )
    generator.generate_dataset(words, samples_per_word=cfg["samples_per_word"])


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def plot_learning_curve(history, save_path="learning_curve.png"):
    """Plot training and validation loss/accuracy."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train Loss", marker="o", linewidth=2)
    axes[0].plot(history["val_loss"], label="Val Loss", marker="s", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("CTC Loss Over Epochs")
    axes[0].legend()
    axes[0].grid()

    if "train_acc" in history:
        axes[1].plot(history["train_acc"], label="Train Acc", marker="o", linewidth=2)
        axes[1].plot(history["val_acc"], label="Val Acc", marker="s", linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_title("Character Accuracy Over Epochs")
        axes[1].legend()
        axes[1].grid()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ Saved learning curve to {save_path}")


def plot_hyperparameter_comparison(results, save_path="hyperparam_comparison.png"):
    """Compare hyperparameters across models."""
    model_names = [r["model"] for r in results]
    val_losses = [r["best_val_loss"] for r in results]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.barh(model_names, val_losses, color="steelblue")

    # Add value labels on bars
    for i, (bar, loss) in enumerate(zip(bars, val_losses)):
        ax.text(loss + 0.01, i, f"{loss:.4f}", va="center")

    ax.set_xlabel("Best Validation Loss")
    ax.set_title("Model Comparison: Hyperparameter Tuning Results")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ Saved hyperparameter comparison to {save_path}")


def plot_model_architecture_comparison(device):
    """Visualize architecture differences."""
    from model.backbones import get_backbone

    backbones = ["simple_cnn", "resnet18", "vgg11"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    for idx, backbone_name in enumerate(backbones):
        backbone = get_backbone(backbone_name, img_channels=1, dropout_rate=0.0)
        x = torch.randn(1, 1, 32, 256).to(device)

        try:
            with torch.no_grad():
                out = backbone(x)
            output_shape = f"Output: {tuple(out.shape)}"
            param_count = sum(p.numel() for p in backbone.parameters())
            param_info = f"Params: {param_count / 1e6:.2f}M"

            axes[idx].text(0.5, 0.7, output_shape, ha="center", fontsize=12, weight="bold")
            axes[idx].text(0.5, 0.5, param_info, ha="center", fontsize=12, weight="bold")
            axes[idx].text(0.5, 0.3, backbone_name.upper(), ha="center", fontsize=14, weight="bold")
            axes[idx].set_xlim(0, 1)
            axes[idx].set_ylim(0, 1)
            axes[idx].axis("off")

        except Exception as e:
            axes[idx].text(0.5, 0.5, f"Error: {str(e)}", ha="center", fontsize=10)
            axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig("model_architecture_comparison.png", dpi=150)
    plt.close()
    print("✓ Saved model architecture comparison to model_architecture_comparison.png")


def visualize_batch_predictions(model, val_loader, charset, device, n_samples=12, save_path="batch_predictions.png"):
    """Visualize predictions on validation samples."""
    model.eval()

    # Get one batch
    imgs, labels_concat, label_lengths = next(iter(val_loader))
    imgs = imgs.to(device)

    with torch.no_grad():
        preds = model(imgs)

    preds_softmax = preds.softmax(2)
    preds_argmax = preds_softmax.argmax(2)  # (T, B)

    # Visualization
    n_show = min(n_samples, imgs.size(0))
    fig, axes = plt.subplots(n_show, 2, figsize=(12, n_show * 2))
    if n_show == 1:
        axes = [axes]

    for idx in range(n_show):
        # Plot image
        img = imgs[idx, 0].cpu().numpy()
        axes[idx][0].imshow(img, cmap="gray")
        axes[idx][0].set_title(f"Sample {idx+1}")
        axes[idx][0].axis("off")

        # Plot prediction probabilities
        pred_probs = preds_softmax[:, idx].cpu().numpy()
        axes[idx][1].plot(pred_probs, linewidth=1)
        axes[idx][1].set_title(f"Confidence per timestep")
        axes[idx][1].set_ylabel("Probability")
        axes[idx][1].grid()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ Saved batch predictions visualization to {save_path}")


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_single_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=50,
    learning_rate=0.001,
    weight_decay=1e-5,
    scheduler_step=15,
    scheduler_gamma=0.5,
    model_name="ocr_model",
    writer=None
):
    """Train a single model configuration."""
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=scheduler_step,
        gamma=scheduler_gamma
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_samples = 0

        for batch_idx, (imgs, labels_concat, label_lengths) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels_concat = labels_concat.to(device)
            label_lengths = label_lengths.to(device)

            preds = model(imgs)
            seq_len, batch_size = preds.size(0), preds.size(1)
            pred_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)

            loss = model.compute_ctc_loss(preds, labels_concat, pred_lengths, label_lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += loss.item() * batch_size
            train_samples += batch_size

        train_loss_avg = train_loss / train_samples

        # Validation
        model.eval()
        val_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for imgs, labels_concat, label_lengths in val_loader:
                imgs = imgs.to(device)
                labels_concat = labels_concat.to(device)
                label_lengths = label_lengths.to(device)

                preds = model(imgs)
                seq_len, batch_size = preds.size(0), preds.size(1)
                pred_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)

                loss = model.compute_ctc_loss(preds, labels_concat, pred_lengths, label_lengths)
                val_loss += loss.item() * batch_size
                val_samples += batch_size

        val_loss_avg = val_loss / val_samples

        history["train_loss"].append(train_loss_avg)
        history["val_loss"].append(val_loss_avg)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_loss={train_loss_avg:.4f}, val_loss={val_loss_avg:.4f}")

        if writer:
            writer.add_scalar(f"{model_name}/train_loss", train_loss_avg, epoch)
            writer.add_scalar(f"{model_name}/val_loss", val_loss_avg, epoch)

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), f"{model_name}_best.pth")

        scheduler.step()

    return history, best_val_loss


# ============================================================
# MAIN TUNING FUNCTION
# ============================================================

def run_hyperparameter_tuning(cfg):
    """Run hyperparameter tuning with different model backbones."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Using device: {device}")

    # Prepare dataset
    prepare_word_dataset(cfg, overwrite=False)

    # Load dataset
    dataset = WordOCRDataset(
        image_dir=cfg["output_dir"],
        img_height=cfg["img_height"],
        img_width=cfg["img_width"]
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    charset = load_charset()
    num_classes = len(charset)

    print(f"✓ Dataset loaded: train={len(train_dataset)}, val={len(val_dataset)}, classes={num_classes}\n")

    # Visualization: Model architecture comparison
    print("📊 Visualizing model architectures...")
    plot_model_architecture_comparison(device)

    # Define hyperparameter grid
    backbones = ["simple_cnn", "resnet18", "vgg11"]
    learning_rates = cfg["hyperparams"]["learning_rates"][:2]  # Reduce for demo
    batch_sizes = cfg["hyperparams"]["batch_sizes"][:2]
    dropout_rates = cfg["hyperparams"]["dropout_rates"][:2]

    results = []
    total_configs = len(backbones) * len(learning_rates) * len(batch_sizes) * len(dropout_rates)
    config_idx = 0

    for backbone, lr, batch_size, dropout_rate in product(backbones, learning_rates, batch_sizes, dropout_rates):
        config_idx += 1
        model_name = f"{backbone}_lr{lr}_bs{batch_size}_drop{dropout_rate}"
        print(f"\n[{config_idx}/{total_configs}] Training: {model_name}")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=WordOCRDataset.collate_fn,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=WordOCRDataset.collate_fn,
            num_workers=0
        )

        # Create model
        model = OCRModelV2(
            num_classes=num_classes,
            backbone_name=backbone,
            img_channels=cfg["num_channels"],
            hidden_size=cfg["hidden_size"],
            dropout_rate=dropout_rate
        ).to(device)

        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {param_count / 1e6:.2f}M")

        # Train
        writer = SummaryWriter(log_dir=f"{cfg['tensorboard_dir']}/{model_name}")

        history, best_val_loss = train_single_model(
            model,
            train_loader,
            val_loader,
            device,
            epochs=cfg["epochs"],
            learning_rate=lr,
            weight_decay=cfg["weight_decay"],
            scheduler_step=cfg["scheduler_step"],
            scheduler_gamma=cfg["scheduler_gamma"],
            model_name=model_name,
            writer=writer
        )

        writer.close()

        # Visualize learning curve
        if cfg.get("plot_training_curves", True):
            plot_learning_curve(history, save_path=f"learning_curve_{model_name}.png")

        # Visualize batch predictions
        if config_idx <= 3:  # Only for first 3 configs to save time
            visualize_batch_predictions(model, val_loader, charset, device, save_path=f"predictions_{model_name}.png")

        results.append({
            "model": model_name,
            "backbone": backbone,
            "lr": lr,
            "batch_size": batch_size,
            "dropout_rate": dropout_rate,
            "best_val_loss": best_val_loss,
            "history": history
        })

    # Final visualizations
    print("\n📊 Generating comparison visualizations...")
    plot_hyperparameter_comparison(results, save_path="hyperparameter_comparison.png")

    # Save results to JSON
    results_json = [
        {k: v for k, v in r.items() if k != "history"} for r in results
    ]
    with open("tuning_results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    print("\n✓ Tuning complete! Results saved to tuning_results.json")
    print("\n📊 Best 5 configurations:")
    sorted_results = sorted(results, key=lambda x: x["best_val_loss"])
    for i, r in enumerate(sorted_results[:5]):
        print(f"  {i+1}. {r['model']}: val_loss={r['best_val_loss']:.4f}")

    return results


if __name__ == "__main__":
    cfg = load_config()
    results = run_hyperparameter_tuning(cfg)

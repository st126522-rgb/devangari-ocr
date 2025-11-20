import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader, random_split
from pathlib import Path

# Add project root to path
proj_root = Path(__file__).parent.parent
sys.path.insert(0, str(proj_root))

from model.ocr_model import OCRModel
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
        print(f"Found {existing} existing images in {output_dir}")
        return

    print("Generating word dataset...")

    # Load cleaned words from dataset
    from datasets import load_dataset
    dataset = load_dataset("Sakonii/nepalitext-language-model-dataset")
    train_texts = dataset["train"]["text"]

    # Clean and extract words
    import re
    words = []
    for text in train_texts:
        if isinstance(text, str):
            # Extract Devanagari words
            matches = re.findall(r"[\u0900-\u097F]+", text)
            words.extend(matches)

    # Sample training words
    words = list(set(words))  # unique
    random.shuffle(words)
    words = words[: cfg["train_samples"]]

    print(f"Using {len(words)} unique words")

    # Generate images
    generator = WordImageGenerator(
        fonts_dir=cfg["fonts_dir"],
        output_dir=output_dir,
        font_size_range=(36, 56),
    )
    generator.generate_dataset(words, samples_per_word=cfg["samples_per_word"])


def train_epoch(model, train_loader, optimizer, device, cfg):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_idx, (imgs, labels_concat, label_lengths) in enumerate(train_loader):
        imgs = imgs.to(device)
        labels_concat = labels_concat.to(device)
        label_lengths = label_lengths.to(device)

        # Forward pass
        preds = model(imgs)  # (T, B, num_classes)
        seq_len, batch_size = preds.size(0), preds.size(1)

        # Compute input lengths (after CNN downsampling)
        pred_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)

        # Compute loss
        loss = model.compute_ctc_loss(preds, labels_concat, pred_lengths, label_lengths)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx + 1}: loss={loss.item():.4f}")

    return total_loss / total_samples


def evaluate(model, val_loader, device, cfg):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for imgs, labels_concat, label_lengths in val_loader:
            imgs = imgs.to(device)
            labels_concat = labels_concat.to(device)
            label_lengths = label_lengths.to(device)

            preds = model(imgs)
            seq_len, batch_size = preds.size(0), preds.size(1)
            pred_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)

            loss = model.compute_ctc_loss(preds, labels_concat, pred_lengths, label_lengths)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def main():
    # Load config
    cfg = load_config()
    charset = load_charset()

    print(f"Config: {cfg}")
    print(f"Charset size: {len(charset)}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare dataset
    prepare_word_dataset(cfg, overwrite=False)

    # Load dataset
    dataset = WordOCRDataset(
        image_dir=cfg["output_dir"],
        img_height=cfg["img_height"],
        img_width=cfg["img_width"]
    )

    # Train / val split (80 / 20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=WordOCRDataset.collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=WordOCRDataset.collate_fn,
        num_workers=0
    )

    # Model
    model = OCRModel(
        num_classes=cfg["num_classes"],
        img_channels=cfg["num_channels"],
        hidden_size=cfg["hidden_size"]
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["scheduler_step"],
        gamma=cfg["scheduler_gamma"]
    )

    # Training loop
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    for epoch in range(cfg["epochs"]):
        print(f"\nEpoch {epoch + 1}/{cfg['epochs']}")

        train_loss = train_epoch(model, train_loader, optimizer, device, cfg)
        val_loss = evaluate(model, val_loader, device, cfg)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

        scheduler.step()

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "ocr_model_best.pth")
            print("Saved best model")

    # Final model
    torch.save(model.state_dict(), "ocr_model.pth")
    print("\nTraining complete. Models saved to ocr_model.pth and ocr_model_best.pth")

    # Plot history
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Train Loss", marker="o")
    plt.plot(history["val_loss"], label="Val Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("OCR Training History")
    plt.legend()
    plt.grid()
    plt.savefig("ocr_training_history.png")
    print("Saved training history to ocr_training_history.png")


if __name__ == "__main__":
    main()

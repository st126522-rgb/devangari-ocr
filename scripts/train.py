import torch
import yaml
from torch.utils.data import DataLoader
from model.ocr_model import OCRModel
from data.pipeline import OCRDataset  # your dataset
from pathlib import Path

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def train():
    cfg = load_config()

    dataset = OCRDataset(cfg["data_root"] + "/images")
    dataloader = DataLoader(dataset,
                            batch_size=cfg["batch_size"],
                            shuffle=True,
                            collate_fn=dataset.collate_fn)

    model = OCRModel(
        num_classes=cfg["num_classes"],
        img_channels=cfg["num_channels"],
        hidden_size=cfg["hidden_size"]
    ).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0

        for imgs, labels, label_lengths in dataloader:

            imgs = imgs.cuda()
            labels = labels.cuda()

            preds = model(imgs)
            pred_lengths = torch.full(size=(imgs.size(0),), fill_value=preds.size(0), dtype=torch.long)

            loss = model.compute_ctc_loss(preds, labels, pred_lengths, label_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{cfg['epochs']}, Loss={total_loss:.3f}")

if __name__ == "__main__":
    train()

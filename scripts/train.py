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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = OCRDataset(cfg["data_root"] + "/images")
    dataloader = DataLoader(dataset,
                            batch_size=cfg["batch_size"],
                            shuffle=True,
                            collate_fn=dataset.collate_fn)

    model = OCRModel(
        num_classes=cfg["num_classes"],
        img_channels=cfg["num_channels"],
        hidden_size=cfg["hidden_size"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0

        for imgs, labels, label_lengths in dataloader:

            imgs = imgs.to(device)
            labels = labels.to(device)

            preds = model(imgs)
            pred_lengths = torch.full(
                size=(imgs.size(0),),
                fill_value=preds.size(0),
                dtype=torch.long,
                device=device,
            )

            loss = model.compute_ctc_loss(preds, labels, pred_lengths, label_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{cfg['epochs']}, Loss={total_loss:.3f}")

    # Save the trained model at the end of training
    torch.save(model.state_dict(), "ocr_model.pth")
    print("Model saved as ocr_model.pth")


if __name__ == "__main__":
    train()

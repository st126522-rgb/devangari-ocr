import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms as T

class LabelEncoder:
    """Converts text → tensor of ints and back."""
    def __init__(self, charset):
        self.charset = sorted(list(charset))
        self.char_to_id = {c: i + 1 for i, c in enumerate(self.charset)}  # 0 = CTC blank
        self.id_to_char = {i + 1: c for i, c in enumerate(self.charset)}

    def encode(self, text):
        return torch.tensor([self.char_to_id[c] for c in text], dtype=torch.long)

    def decode(self, ids):
        return "".join([self.id_to_char[i] for i in ids if i != 0])


class OCRDataset(Dataset):
    def __init__(self, root_dir, img_height=32, img_width=128):
        self.root = root_dir
        self.samples = []

        # Collect image + label pairs
        for f in os.listdir(root_dir):
            if f.endswith(".png"):
                img = os.path.join(root_dir, f)
                txt = img.replace(".png", ".gt.txt")
                if os.path.exists(txt):
                    self.samples.append((img, txt))

        self.samples.sort()

        self.transform = T.Compose([
            T.Grayscale(),
            T.Resize((img_height, img_width)),
            T.ToTensor(),
        ])

        # Build charset dynamically
        charset = set()
        for _, txt_path in self.samples:
            with open(txt_path, "r", encoding="utf-8") as f:
                charset.update(list(f.read().strip()))

        self.encoder = LabelEncoder(charset)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, txt_path = self.samples[idx]

        img = Image.open(img_path).convert("L")
        img = self.transform(img)

        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        encoded = self.encoder.encode(text)
        return img, encoded, len(encoded)

    def collate_fn(self, batch):
        imgs = torch.stack([b[0] for b in batch])
        labels = torch.cat([b[1] for b in batch])
        label_lengths = torch.tensor([b[2] for b in batch], dtype=torch.long)
        return imgs, labels, label_lengths

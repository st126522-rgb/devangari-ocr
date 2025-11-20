import os
import glob
import random
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms
import freetype
import uharfbuzz as hb
import cv2


class WordImageGenerator:
    """Generate synthetic word images using HarfBuzz + FreeType."""

    def __init__(
        self,
        fonts_dir="fonts",
        output_dir="data/word_images",
        font_size_range=(36, 56),
        max_image_size=1024,
    ):
        self.fonts = glob.glob(os.path.join(fonts_dir, "**/*.ttf"), recursive=True)
        if not self.fonts:
            raise ValueError(f"No fonts found in {fonts_dir}")

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.font_size_range = font_size_range
        self.MAX_SIZE = max_image_size

    # -----------------------------------------------
    def _clamp_image_size(self, img):
        w, h = img.size
        if w > self.MAX_SIZE or h > self.MAX_SIZE:
            img.thumbnail((self.MAX_SIZE, self.MAX_SIZE), Image.LANCZOS)
        return img

    # -----------------------------------------------
    def render_word_image(self, text, padding=20):
        """Render a single word using HarfBuzz shaping."""
        font_path = random.choice(self.fonts)
        font_size = random.randint(*self.font_size_range)

        # FreeType setup
        face = freetype.Face(font_path)
        face.set_char_size(font_size * 64)

        # HarfBuzz shaping
        hb_blob = hb.Blob.from_file_path(font_path)
        hb_face = hb.Face(hb_blob, 0)
        hb_font = hb.Font(hb_face)
        hb_font.scale = (face.size.ascender, face.size.ascender)

        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(hb_font, buf)

        infos = buf.glyph_infos
        positions = buf.glyph_positions

        width = sum(pos.x_advance for pos in positions) // 64 + 2 * padding
        height = font_size + 2 * padding

        # Create image
        bg_color = random.choice(["white", "lightgray"])
        if bg_color == "white":
            img = Image.new("RGB", (width, height), "white")
        else:
            arr = np.random.randint(200, 255, (height, width, 3), dtype=np.uint8)
            img = Image.fromarray(arr)

        x, y = padding, padding + font_size

        # Render glyphs
        for info, pos in zip(infos, positions):
            glyph_index = info.codepoint
            face.load_glyph(glyph_index, freetype.FT_LOAD_RENDER)
            bitmap = face.glyph.bitmap
            top = face.glyph.bitmap_top
            left = face.glyph.bitmap_left

            if bitmap.width > 0 and bitmap.rows > 0:
                glyph_img = Image.frombytes("L", (bitmap.width, bitmap.rows), bytes(bitmap.buffer))
                colored_glyph = Image.new("RGB", glyph_img.size, "black")
                img.paste(colored_glyph, (int(x + left), int(y - top)), glyph_img)

            x += pos.x_advance / 64
            y -= pos.y_advance / 64

        img = self._clamp_image_size(img)

        # Random blur
        if random.random() < 0.4:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        # Random rotation
        if random.random() < 0.3:
            angle = random.randint(-5, 5)
            img = img.rotate(angle, expand=True, fillcolor="white")
            img = self._clamp_image_size(img)

        # Random distortion
        if random.random() < 0.3:
            img = self.perspective_distortion(img)

        # Random noise
        if random.random() < 0.3:
            img = self.add_noise(img)

        return img

    # -----------------------------------------------
    def perspective_distortion(self, img):
        img = self._clamp_image_size(img)
        w, h = img.size
        arr = np.array(img)
        shift = min(w, h) * 0.08

        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([
            [random.uniform(-shift, shift), random.uniform(-shift, shift)],
            [w + random.uniform(-shift, shift), random.uniform(-shift, shift)],
            [random.uniform(-shift, shift), h + random.uniform(-shift, shift)],
            [w + random.uniform(-shift, shift), h + random.uniform(-shift, shift)],
        ])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(
            arr, matrix, (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        return Image.fromarray(warped)

    # -----------------------------------------------
    def add_noise(self, img):
        arr = np.array(img).astype(np.float32)

        if random.random() < 0.5:
            arr += np.random.normal(0, 8, arr.shape)

        if random.random() < 0.4:
            amount = 0.015
            num_salt = int(arr.size * amount * 0.5)
            num_pepper = int(arr.size * amount * 0.5)

            coords = [np.random.randint(0, i - 1, num_salt) for i in arr.shape]
            arr[tuple(coords)] = 255

            coords = [np.random.randint(0, i - 1, num_pepper) for i in arr.shape]
            arr[tuple(coords)] = 0

        arr = np.clip(arr, 0, 255)
        return Image.fromarray(arr.astype(np.uint8))

    # -----------------------------------------------
    def generate_dataset(self, words, samples_per_word=5):
        """Generate images for a list of words."""
        idx = 0
        for word_idx, word in enumerate(words):
            for var in range(samples_per_word):
                img = self.render_word_image(word)

                image_path = os.path.join(self.output_dir, f"{idx:06d}.png")
                label_path = os.path.join(self.output_dir, f"{idx:06d}.txt")

                img_gray = img.convert("L")
                img_gray.save(image_path)

                with open(label_path, "w", encoding="utf-8") as f:
                    f.write(word)

                if (idx + 1) % 500 == 0:
                    print(f"Generated {idx + 1} images...")

                idx += 1

        print(f"Total generated: {idx} images")


# -----------------------------------------------
# DATASET CLASS
# -----------------------------------------------

class WordOCRDataset(Dataset):
    """Load word images + labels for OCR training."""

    def __init__(self, image_dir, img_height=32, img_width=256):
        self.image_dir = image_dir
        self.img_height = img_height
        self.img_width = img_width

        # Load charset
        self.idx_to_char = self._load_charset()
        self.char_to_idx = {c: i + 1 for i, c in enumerate(self.idx_to_char)}

        # Load samples
        self.samples = self._load_samples()

        # Transforms
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    # -----------------------------------------------
    def _load_charset(self):
        charset_path = "charset.txt"
        with open(charset_path, "r", encoding="utf-8") as f:
            return list(f.read().strip())

    # -----------------------------------------------
    def _load_samples(self):
        samples = []
        for file in sorted(os.listdir(self.image_dir)):
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(self.image_dir, file)
                txt_path = os.path.splitext(img_path)[0] + ".txt"

                if os.path.exists(txt_path):
                    samples.append((img_path, txt_path))

        return samples

    # -----------------------------------------------
    def encode_text(self, text):
        """Convert text to character indices."""
        encoded = []
        for ch in text.strip():
            if ch in self.char_to_idx:
                encoded.append(self.char_to_idx[ch])
        return encoded if encoded else [1]  # fallback to first char

    # -----------------------------------------------
    def __len__(self):
        return len(self.samples)

    # -----------------------------------------------
    def __getitem__(self, idx):
        img_path, txt_path = self.samples[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # Resize to fixed height, pad width
        new_w = int(w * (self.img_height / h))
        img = img.resize((new_w, self.img_height), Image.BICUBIC)

        if new_w < self.img_width:
            new_img = Image.new("RGB", (self.img_width, self.img_height), (255, 255, 255))
            new_img.paste(img, (0, 0))
            img = new_img
        else:
            img = img.crop((0, 0, self.img_width, self.img_height))

        img = self.transform(img)

        # Load label
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        label = self.encode_text(text)
        label_length = len(label)

        return img, torch.tensor(label, dtype=torch.long), label_length

    # -----------------------------------------------
    @staticmethod
    def collate_fn(batch):
        """Collate for CTC training."""
        imgs = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        lengths = [item[2] for item in batch]

        imgs = torch.stack(imgs, 0)
        labels_concat = torch.cat(labels)
        lengths = torch.tensor(lengths, dtype=torch.long)

        return imgs, labels_concat, lengths

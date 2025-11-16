import os
from pathlib import Path

def build_charset(image_dir, output="charset.txt"):
    charset = set()

    for file in os.listdir(image_dir):
        if file.endswith(".gt.txt") or file.endswith(".txt"):
            txt_path = os.path.join(image_dir, file)
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                for ch in text:
                    charset.add(ch)

    charset = sorted(list(charset))

    with open(output, "w", encoding="utf-8") as f:
        for ch in charset:
            f.write(ch + "\n")

    print(f"✔ Charset saved to {output}")
    print(f"✔ Total characters: {len(charset)}")

if __name__ == "__main__":
    build_charset("data/images")

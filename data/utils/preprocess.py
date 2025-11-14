"""
Preprocessing helpers: tokenization mapping and image preprocessing stubs.
"""

def build_vocab_from_texts(texts, min_freq=1):
    # implement character-level vocab builder for Devanagari
    chars = {}
    for t in texts:
        for c in t:
            chars[c] = chars.get(c, 0) + 1
    vocab = {c:i+1 for i,(c,f) in enumerate(chars.items()) if f>=min_freq}
    vocab["<pad>"] = 0
    return vocab

def image_preprocess(img):
    # placeholder: resize/normalize image (PIL / OpenCV)
    return img

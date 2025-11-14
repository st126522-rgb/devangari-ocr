"""
Image augmentation helpers. Replace with real transforms as needed.
"""
from PIL import Image, ImageOps

def random_rotate(img, degrees=5):
    return img.rotate(degrees)

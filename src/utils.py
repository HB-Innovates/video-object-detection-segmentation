import numpy as np
import cv2
from torchvision import transforms

def random_crop(image, crop_size):
    """
    Randomly crop the input image to the specified size.
    Args:
        image (np.ndarray): Input image.
        crop_size (tuple): (height, width) of the crop.
    Returns:
        np.ndarray: Cropped image.
    """
    h, w = image.shape[:2]
    ch, cw = crop_size
    if h < ch or w < cw:
        raise ValueError("Crop size must be smaller than image size.")
    top = np.random.randint(0, h - ch + 1)
    left = np.random.randint(0, w - cw + 1)
    return image[top:top + ch, left:left + cw]

def color_jitter(image, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
    """
    Apply color jitter to the input image.
    Args:
        image (np.ndarray): Input image (BGR).
        brightness, contrast, saturation, hue: Jitter parameters.
    Returns:
        np.ndarray: Jittered image.
    """
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = transforms.ToPILImage()(img_rgb)
    jitter = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    pil_img = jitter(pil_img)
    img_jittered = np.array(pil_img)
    return cv2.cvtColor(img_jittered, cv2.COLOR_RGB2BGR)

def load_image(image_path):
    """
    Load an image from disk.
    Args:
        image_path (str): Path to image file.
    Returns:
        np.ndarray: Loaded image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image
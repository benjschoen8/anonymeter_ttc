import os
import numpy as np
import pandas as pd
from PIL import Image

def image_to_row(image_path, file, grayscale=True):
    """Convert one image to a row with name, type, size, and flattened pixel values."""
    img = Image.open(image_path)
    if grayscale:
        img = img.convert('L')  # Grayscale
    else:
        img = img.convert('RGB')

    pixels = np.array(img)

    # Flatten pixel matrix
    flattened = pixels.flatten().tolist()

    # Extract name and type
    name, ext = os.path.splitext(file)
    ext = ext.lstrip('.')  # Remove leading dot

    # Size string (e.g., "32x32")
    size_str = f"{img.width}x{img.height}"

    return {
        'name': name,
        'type': ext,
        'size': size_str,
        'pixels': flattened
    }

def process_folder(folder_path, output_csv, image_size=(32, 32), grayscale=True):
    """Convert a folder of images into a CSV with: name, type, size, flattened_pixels."""
    rows = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                full_path = os.path.join(root, file)
                row = image_to_row(full_path, file, grayscale=grayscale)
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"[✓] Saved {len(df)} images to {output_csv} with columns: name, type, size, flattened_pixels")

if __name__ == "__main__":
    folder = "dataset"  # Change to your folder path
    output = "images.csv"
    process_folder(folder, output, image_size=(32, 32), grayscale=True)


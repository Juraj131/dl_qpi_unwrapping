import os
import numpy as np
import tifffile
from custom_reconstruction_tiff_create import fit_and_subtract_background

def subtract_background_batch(input_dir, output_dir, order=1):
    """
    Aplikuje fitovanie a odčítanie pozadia (polynóm) na všetky obrázky v zložke.
    Výsledky uloží do output_dir bez zmeny pôvodných dát.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(input_dir) if f.endswith(".tiff") or f.endswith(".tif")]

    print(f"\n--- Spúšťam odčítanie pozadia ---")
    print(f"Počet obrázkov na spracovanie: {len(image_files)}")
    print(f"Polynómový stupeň: {order}")

    for fname in image_files:
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)

        img = tifffile.imread(input_path).astype(np.float32)
        corrected, _ = fit_and_subtract_background(img, order=order)

        tifffile.imwrite(output_path, corrected.astype(np.float32))
        print(f"  → spracované: {fname}")

    print(f"\nVýstupné obrázky uložené do: {output_dir}")

# Príklad volania (prispôsob podľa seba):
subtract_background_batch("unwrapped", "unwrapped_bgsubtracted", order=3)

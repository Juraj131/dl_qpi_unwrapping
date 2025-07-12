import os
import re
import random
import shutil

def count_images_by_r_number(input_dir):
    """
    Pre každý obrázok v input_dir zistí dvojčíslie za znakom 'r' na začiatku názvu
    a spočíta, koľko obrázkov patrí ku každému dvojčíslu okrem r12, r13, r14.
    Výstup: zoznam dvojíc (r??, počet), napr. [('r01', 15), ('r02', 20), ...]
    """
    counts = {}
    exclude = {'12', '13', '14'}
    for fname in os.listdir(input_dir):
        match = re.match(r"r(\d{2})", fname)
        if match and match.group(1) not in exclude:
            r_number = f"r{match.group(1)}"
            counts[r_number] = counts.get(r_number, 0) + 1
    return sorted(counts.items())

def create_filtered_dataset(input_dir, output_dir):
    """
    Skopíruje všetky obrázky z input_dir do output_dir okrem tých, ktoré majú r12, r13 alebo r14.
    """
    os.makedirs(output_dir, exist_ok=True)
    exclude = {'12', '13', '14'}
    for fname in os.listdir(input_dir):
        match = re.match(r"r(\d{2})", fname)
        if match and match.group(1) not in exclude:
            src = os.path.join(input_dir, fname)
            dst = os.path.join(output_dir, fname)
            shutil.copy2(src, dst)
    print(f"Vytvorený filtrovaný dataset bez r12, r13, r14 v: {output_dir}")

def split_dataset_by_r_number(input_dir, output_base, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Rozdelí obrázky podľa r?? rovnomerne do train/val/test priečinkov, ignoruje r12, r13, r14.
    """
    random.seed(seed)
    exclude = {'12', '13', '14'}
    # Zozbieraj obrázky podľa r??
    r_groups = {}
    for fname in os.listdir(input_dir):
        match = re.match(r"r(\d{2})", fname)
        if match and match.group(1) not in exclude:
            r_number = f"r{match.group(1)}"
            r_groups.setdefault(r_number, []).append(fname)
    # Priprav výstupné priečinky
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_base, split), exist_ok=True)
    # Rozdeľuj a kopíruj
    for r_number, files in r_groups.items():
        random.shuffle(files)
        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        splits = (
            ('train', files[:n_train]),
            ('val', files[n_train:n_train+n_val]),
            ('test', files[n_train+n_val:])
        )
        for split_name, split_files in splits:
            for fname in split_files:
                src = os.path.join(input_dir, fname)
                dst = os.path.join(output_base, split_name, fname)
                shutil.copy2(src, dst)
    print("Rozdelenie hotové.")

# Príklad použitia:
vysledok = count_images_by_r_number("ch1_cropped_resized")
print(vysledok)
create_filtered_dataset("ch1_cropped_resized", "ch1_filtered_no_r12_13_14")
split_dataset_by_r_number("ch1_cropped_resized", "ch1_split", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
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
    exclude = {'5', '12', '13', '14'}
    for fname in os.listdir(input_dir):
        match = re.match(r"r(\d{2})", fname)
        if match and match.group(1) not in exclude:
            r_number = f"r{match.group(1)}"
            counts[r_number] = counts.get(r_number, 0) + 1
    return sorted(counts.items())

def create_filtered_dataset(input_dir, output_dir, r_numbers_to_exclude=None):
    """
    Skopíruje všetky TIFF obrázky z input_dir do output_dir.
    Štandardne preskakuje obrázky s r-číslami zodpovedajúcimi 5, 12, 13, 14 (t.j. r05, r12, r13, r14).
    Tento zoznam je možné upraviť pomocou parametra r_numbers_to_exclude (napr. [5, 12] alebo [] pre žiadne vylúčenie).
    Čísla sú interpretované ako dvojciferné (napr. 5 sa stane '05' pre porovnanie s názvom súboru ako r05...).
    """
    if r_numbers_to_exclude is None:
        r_numbers_to_exclude = [5, 12, 13, 14]  # Predvolený zoznam na vylúčenie

    os.makedirs(output_dir, exist_ok=True)
    
    # Formátovanie čísel na vylúčenie na dvojciferné reťazce (napr. 5 -> "05", 12 -> "12")
    exclude_formatted = {f"{int(num):02d}" for num in r_numbers_to_exclude}
    
    excluded_files_count = 0
    copied_files_count = 0
    processed_files_count = 0

    for fname in os.listdir(input_dir):
        # Spracuj len .tif alebo .tiff súbory
        if not (fname.lower().endswith('.tif') or fname.lower().endswith('.tiff')):
            continue

        in_path = os.path.join(input_dir, fname)
        # Uisti sa, že ide o súbor
        if not os.path.isfile(in_path):
            continue
        
        processed_files_count +=1
        match = re.match(r"r(\d{2})", fname)
        
        if match and match.group(1) in exclude_formatted:
            excluded_files_count += 1
            continue  # Preskočí kopírovanie, ak r-číslo je v zozname na vylúčenie
        
        # Ak súbor nie je na vylúčenie, skopíruje sa
        dst = os.path.join(output_dir, fname)
        shutil.copy2(in_path, dst)
        copied_files_count += 1
            
    print(f"Spracovanie adresára '{input_dir}' dokončené.")
    print(f"Výstupný adresár: {output_dir}")
    print(f"Celkovo skontrolovaných .tif/.tiff súborov: {processed_files_count}")
    print(f"Počet skopírovaných súborov: {copied_files_count}")

    if exclude_formatted:
        excluded_r_display = ", ".join(f"r{s}" for s in sorted(list(exclude_formatted)))
        print(f"Boli nastavené na vylúčenie obrázky s r-číslami: {excluded_r_display if excluded_r_display else 'žiadne'}.")
        print(f"Počet súborov preskočených na základe r-čísla: {excluded_files_count}")
    else:
        print("Neboli špecifikované žiadne r-čísla na vylúčenie.")

def split_dataset_by_r_number(input_dir, output_base, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Rozdelí obrázky podľa r?? rovnomerne do train/val/test priečinkov, ignoruje r12, r13, r14.
    """
    random.seed(seed)
    exclude = {'5', '12', '13', '14'}
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
vysledok = count_images_by_r_number("collected_chanels/ch4_cropped_resized")
print(vysledok)

# Pôvodné volanie by teraz malo fungovať podľa očakávania (vylúči r05, r12, r13, r14)
create_filtered_dataset("collected_chanels/ch4_cropped_resized", "ch4_filtered_default_exclusions")

# Príklad: Vylúčenie iba r05 a r07
# create_filtered_dataset("collected_chanels/ch1_cropped_resized", "ch1_filtered_custom", r_numbers_to_exclude=[5, 7])

# Príklad: Žiadne vylúčenie na základe r-čísla
# create_filtered_dataset("collected_chanels/ch1_cropped_resized", "ch1_filtered_no_r_exclusions", r_numbers_to_exclude=[])

#split_dataset_by_r_number("ch1_cropped_resized", "ch1_split", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
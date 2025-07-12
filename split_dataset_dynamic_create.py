import os
import shutil
import random
import re
import glob
import time
import numpy as np
from skimage import io
from skimage.filters import gaussian
import tifffile
from collections import defaultdict

# --- Pomocná funkcia na generovanie náhodných hodnôt ---
def get_random_or_fixed(param_value, is_integer=False, allow_float_for_int=False):
    if isinstance(param_value, (list, tuple)) and len(param_value) == 2:
        min_val, max_val = param_value
        if min_val > max_val: # Zabezpečenie, aby min_val nebol väčší ako max_val
            min_val_temp = min_val
            min_val = max_val
            max_val = min_val_temp
        if is_integer:
            low = int(round(min_val))
            high = int(round(max_val))
            if high < low: high = low # Ak by zaokrúhlenie spôsobilo problém
            
            # Použijeme priamo int() na pôvodné hodnoty ak sú už celé čísla a allow_float_for_int je False
            if not allow_float_for_int and isinstance(min_val, int) and isinstance(max_val, int):
                return np.random.randint(min_val, max_val + 1)
            else: # Inak použijeme zaokrúhlené hodnoty
                return int(np.random.randint(low, high + 1))
        else: # Pre float hodnoty
            return np.random.uniform(min_val, max_val)
    return param_value # Ak to nie je list/tuple, vráti pôvodnú hodnotu (napr. fixná hodnota)

# --- Funkcia na zabalenie fázy ---
def wrap_phase(img):
    return (img + np.pi) % (2 * np.pi) - np.pi

# --- Funkcia na generovanie kubického pozadia (zostáva rovnaká ako vo vašom pôvodnom kóde) ---
def generate_cubic_background(shape, coeff_stats, scale=1.0, amplify_ab=1.0, n_strips=6, tilt_angle_deg=0.0):
    H, W = shape
    y_idxs, x_idxs = np.indices((H, W))

    slope_y = (n_strips * 2 * np.pi) / H
    tilt_angle_rad = np.deg2rad(tilt_angle_deg)
    slope_x = slope_y * np.tan(tilt_angle_rad)
    linear_grad = slope_y * y_idxs + slope_x * x_idxs

    x_norm = np.linspace(0, 1, W)
    a_val = np.random.normal(coeff_stats[0][0], coeff_stats[0][1] * scale)
    b_val = np.random.normal(coeff_stats[1][0], coeff_stats[1][1] * scale)
    c_val = np.random.normal(coeff_stats[2][0], coeff_stats[2][1] * scale)
    d_val = np.random.normal(coeff_stats[3][0], coeff_stats[3][1] * scale)
    
    a_val *= amplify_ab
    b_val *= amplify_ab
    
    poly1d = a_val * x_norm**3 + b_val * x_norm**2 + c_val * x_norm + d_val
    poly2d = np.tile(poly1d, (H, 1))
    background = linear_grad + poly2d
    return background, dict(a=a_val, b=b_val, c=c_val, d=d_val, n_strips_actual=n_strips,
                            slope_y=slope_y, slope_x=slope_x, tilt_angle_deg=tilt_angle_deg)

# --- Hlavná simulačná funkcia (vracia dáta) ---
def generate_simulation_pair_from_source_np(
    source_image_np, # Vstup je už načítaný a normalizovaný NumPy array
    param_ranges_config, # Slovník obsahujúci všetky rozsahy parametrov
    amplify_ab_value, # Fixná hodnota pre amplify_ab
    verbose_simulation=False
):
    # Získanie náhodných parametrov z definovaných rozsahov
    n_strips = get_random_or_fixed(param_ranges_config["n_strips_param"], is_integer=True, allow_float_for_int=True)
    original_image_influence = get_random_or_fixed(param_ranges_config["original_image_influence_param"])
    phase_noise_std = get_random_or_fixed(param_ranges_config["phase_noise_std_param"])
    smooth_original_image_sigma = get_random_or_fixed(param_ranges_config["smooth_original_image_sigma_param"])
    poly_scale = get_random_or_fixed(param_ranges_config["poly_scale_param"])
    CURVATURE_AMPLITUDE = get_random_or_fixed(param_ranges_config["CURVATURE_AMPLITUDE_param"])
    background_d_offset = get_random_or_fixed(param_ranges_config["background_offset_d_param"])
    tilt_angle_deg = get_random_or_fixed(param_ranges_config["tilt_angle_deg_param"])

    coeff_stats_for_bg_generation = [
        (0.0, 0.3 * CURVATURE_AMPLITUDE),
        (-4.0 * CURVATURE_AMPLITUDE, 0.3 * CURVATURE_AMPLITUDE),
        (+4.0 * CURVATURE_AMPLITUDE, 0.3 * CURVATURE_AMPLITUDE),
        (background_d_offset, 2.0) 
    ]

    if verbose_simulation:
        print(f"    Sim Params: n_strips={n_strips}, obj_influence={original_image_influence:.2f}, noise_std={phase_noise_std:.3f}, smooth_sigma={smooth_original_image_sigma:.2f}")
        print(f"                poly_scale={poly_scale:.3f}, curv_amp={CURVATURE_AMPLITUDE:.2f}, d_offset={background_d_offset:.2f}, tilt_angle={tilt_angle_deg:.2f} deg, amplify_ab={amplify_ab_value:.2f}")

    if smooth_original_image_sigma > 0 and original_image_influence > 0:
        img_phase_obj_base = gaussian(source_image_np, sigma=smooth_original_image_sigma, preserve_range=True, channel_axis=None)
    else:
        img_phase_obj_base = source_image_np
    object_phase_contribution = img_phase_obj_base * (2 * np.pi)

    generated_background, _ = generate_cubic_background(
        source_image_np.shape, 
        coeff_stats_for_bg_generation, 
        scale=poly_scale, 
        amplify_ab=amplify_ab_value, 
        n_strips=n_strips,
        tilt_angle_deg=tilt_angle_deg
    )
    
    unwrapped_phase = (object_phase_contribution * original_image_influence) + \
                      (generated_background * (1.0 - original_image_influence))

    if phase_noise_std > 0:
        unwrapped_phase += np.random.normal(0, phase_noise_std, size=source_image_np.shape)
    wrapped_phase = wrap_phase(unwrapped_phase)
    return unwrapped_phase.astype(np.float32), wrapped_phase.astype(np.float32)

# --- Hlavná funkcia pre vytvorenie a rozdelenie datasetu ---
def create_final_datasets_stratified(
    source_channel_dirs, 
    output_base_dir,
    simulation_param_ranges_config,
    amplify_ab_config_value,
    ref_train_percentage_of_total=0.1, 
    train_ratio_of_remaining=0.7, 
    valid_ratio_of_remaining=0.15,
    random_seed=42
):
    random.seed(random_seed)
    np.random.seed(random_seed)

    dataset_names = {
        "dynamic_train_source": "train_dataset_source_for_dynamic_generation", # Zdrojové pre dynamický tréning
        "static_ref_train": "static_ref_train_dataset",    # Na výpočet normalizačných štatistík
        "static_valid": "static_valid_dataset",             # Statický validačný
        "static_test": "static_test_dataset"                # Statický testovací
    }
    image_subfolders = ["images", "labels"] # Pre sady obsahujúce páry

    if os.path.exists(output_base_dir):
        print(f"UPOZORNENIE: Výstupný adresár '{output_base_dir}' už existuje. Obsah môže byť prepísaný.")
    # Vytvorenie hlavného výstupného adresára a podadresárov
    for dir_name_suffix in dataset_names.values():
        path_to_create = os.path.join(output_base_dir, dir_name_suffix)
        if "source" in dir_name_suffix: # Pre zdrojové dáta len jeden podadresár 'images'
             os.makedirs(os.path.join(path_to_create, "images"), exist_ok=True)
        else: # Pre statické sady 'images' a 'labels'
            for subfolder in image_subfolders:
                os.makedirs(os.path.join(path_to_create, subfolder), exist_ok=True)
    print(f"Výstupné adresáre pripravené v: {output_base_dir}")

    # 1. Načítanie všetkých zdrojových súborov, rozdelených podľa kanálov
    all_source_files_by_channel = {}
    total_source_files_count = 0
    channel_names_ordered = sorted(list(source_channel_dirs.keys())) # Pre konzistentné poradie

    print("\n--- KROK 1: Načítavanie zdrojových súborov z kanálov ---")
    for ch_name in channel_names_ordered:
        ch_path = source_channel_dirs[ch_name]
        if not os.path.isdir(ch_path):
            print(f"  CHYBA: Adresár pre kanál {ch_name} neexistuje: {ch_path}")
            all_source_files_by_channel[ch_name] = []
            continue
        files = sorted(glob.glob(os.path.join(ch_path, "*.tif*"))) # *.tif* pokryje .tif aj .tiff
        if not files:
            print(f"  UPOZORNENIE: Nenašli sa žiadne TIFF súbory v {ch_path} pre kanál {ch_name}.")
            all_source_files_by_channel[ch_name] = []
        else:
            random.shuffle(files) # Zamiešame súbory v rámci každého kanála pred delením
            all_source_files_by_channel[ch_name] = files
            total_source_files_count += len(files)
            print(f"  Kanál {ch_name}: Nájdených {len(files)} zdrojových súborov.")
    
    if total_source_files_count == 0:
        print("CHYBA: Nenašli sa žiadne zdrojové súbory na spracovanie. Skript končí.")
        return

    # 2. Vyčlenenie zdrojových súborov pre STATICKÝ REFERENČNÝ TRÉNINGOVÝ DATASET
    #    so zachovaním pomerného zastúpenia kanálov
    print(f"\n--- KROK 2: Vyčleňovanie zdrojových súborov pre statický referenčný tréningový dataset ({ref_train_percentage_of_total*100:.0f}% z celku) ---")
    source_files_for_ref_train_generation = [] # Bude [(cesta, kanál), ...]
    remaining_files_by_channel_after_ref = defaultdict(list) # Zvyšok pre hlavné delenie
    
    for ch_name in channel_names_ordered:
        files_in_channel = all_source_files_by_channel.get(ch_name, [])
        if not files_in_channel: continue
        
        num_to_take_for_ref = int(round(ref_train_percentage_of_total * len(files_in_channel)))
        # Zabezpečenie, aby sa zobral aspoň jeden, ak kanál nie je prázdny a percento by dalo 0
        if num_to_take_for_ref == 0 and len(files_in_channel) > 0 and ref_train_percentage_of_total > 0: 
            num_to_take_for_ref = 1
        # Zabezpečenie, aby sme nezobrali viac, ako je k dispozícii
        num_to_take_for_ref = min(num_to_take_for_ref, len(files_in_channel))

        source_files_for_ref_train_generation.extend([(f, ch_name) for f in files_in_channel[:num_to_take_for_ref]])
        remaining_files_by_channel_after_ref[ch_name].extend(files_in_channel[num_to_take_for_ref:])
        print(f"  Z kanála {ch_name}: vyčlenených {num_to_take_for_ref} pre ref. tréning. Zostalo: {len(files_in_channel[num_to_take_for_ref:])}")
    
    print(f"  Celkovo súborov pre referenčný tréningový set: {len(source_files_for_ref_train_generation)}")

    # 3. Rozdelenie ZVYŠNÝCH zdrojových obrázkov na dynamický tréningový, statický validačný a statický testovací set
    #    so zachovaním pomerného zastúpenia kanálov.
    print(f"\n--- KROK 3: Rozdeľovanie zvyšných zdrojových súborov na dynamický tréning, validáciu a test ({train_ratio_of_remaining*100:.0f}%/{valid_ratio_of_remaining*100:.0f}%/zvyšok) ---")
    
    source_files_for_dynamic_train_final = []
    source_files_for_static_valid_final = []
    source_files_for_static_test_final = []

    for ch_name in channel_names_ordered:
        remaining_files = remaining_files_by_channel_after_ref.get(ch_name, [])
        if not remaining_files: 
            print(f"  Kanál {ch_name}: Žiadne zvyšné súbory na rozdelenie.")
            continue
        
        # random.shuffle(remaining_files) # Už sú zamiešané na začiatku v rámci kanála
        
        total_remaining_channel = len(remaining_files)
        train_count_ch = int(train_ratio_of_remaining * total_remaining_channel)
        valid_count_ch = int(valid_ratio_of_remaining * total_remaining_channel)
        test_count_ch = total_remaining_channel - train_count_ch - valid_count_ch

        source_files_for_dynamic_train_final.extend([(f, ch_name) for f in remaining_files[:train_count_ch]])
        source_files_for_static_valid_final.extend([(f, ch_name) for f in remaining_files[train_count_ch : train_count_ch + valid_count_ch]])
        source_files_for_static_test_final.extend([(f, ch_name) for f in remaining_files[train_count_ch + valid_count_ch:]])
        print(f"  Z kanála {ch_name} (zo zvyšku): {train_count_ch} pre dyn.tréning, {valid_count_ch} pre valid., {test_count_ch} pre test.")

    # Globálne zamiešanie finálnych zoznamov (pre náhodné poradie pri kopírovaní/generovaní)
    random.shuffle(source_files_for_dynamic_train_final)
    random.shuffle(source_files_for_static_valid_final)
    random.shuffle(source_files_for_static_test_final)

    print(f"\nFinálne rozdelenie zdrojových súborov (po vyčlenení pre ref. tréning):")
    print(f"  Pre dynamický tréning: {len(source_files_for_dynamic_train_final)} súborov")
    print(f"  Pre statickú validáciu: {len(source_files_for_static_valid_final)} súborov")
    print(f"  Pre statické testovanie: {len(source_files_for_static_test_final)} súborov")

    # 4. Kopírovanie zdrojových súborov pre dynamický tréning
    print(f"\n--- KROK 4: Kopírovanie zdrojových súborov pre dynamický tréning do '{dataset_names['dynamic_train_source']}/images' ---")
    dynamic_train_source_output_dir = os.path.join(output_base_dir, dataset_names['dynamic_train_source'], "images")
    for source_path, _ in source_files_for_dynamic_train_final:
        shutil.copy2(source_path, os.path.join(dynamic_train_source_output_dir, os.path.basename(source_path)))
    print(f"  Skopírovaných {len(source_files_for_dynamic_train_final)} súborov.")
    
    # 5. Generovanie statických datasetov (ref_train, valid, test)
    print(f"\n--- KROK 5: Generovanie statických datasetov (referenčný tréningový, validačný, testovací) ---")
    global_static_id_counter = 0 
    final_set_counts = defaultdict(int)
    final_channel_distribution = defaultdict(lambda: defaultdict(int))

    for set_type_key, source_file_list_for_current_set in [ # Upravený názov premennej
        ("static_ref_train", source_files_for_ref_train_generation), # Použijeme správny zoznam
        ("static_valid", source_files_for_static_valid_final),
        ("static_test", source_files_for_static_test_final)
    ]:
        print(f"  Generujem {dataset_names[set_type_key]}...")
        generated_count_for_this_set = 0 # Lokálny čítač pre aktuálny set
        for source_path, ch_name_for_file in source_file_list_for_current_set:
            try:
                img_np = io.imread(source_path).astype(np.float32)
                img_min_val, img_max_val = img_np.min(), img_np.max()
                source_img_norm = (img_np - img_min_val) / (img_max_val - img_min_val) if img_max_val > img_min_val else np.zeros_like(img_np)
                
                unwrapped, wrapped = generate_simulation_pair_from_source_np(
                    source_img_norm, 
                    simulation_param_ranges_config, 
                    amplify_ab_config_value, 
                    verbose_simulation=False 
                )
                
                id_str = f"{global_static_id_counter:05d}"
                tifffile.imwrite(os.path.join(output_base_dir, dataset_names[set_type_key], "images", f"wrappedbg_{id_str}.tiff"), wrapped)
                tifffile.imwrite(os.path.join(output_base_dir, dataset_names[set_type_key], "labels", f"unwrapped_{id_str}.tiff"), unwrapped)
                global_static_id_counter += 1
                generated_count_for_this_set += 1
                final_set_counts[set_type_key] +=1
                final_channel_distribution[set_type_key][ch_name_for_file] +=1

            except Exception as e:
                print(f"    Chyba pri generovaní páru pre {set_type_key} z {source_path}: {e}")
        print(f"    Vygenerovaných {generated_count_for_this_set} párov pre {dataset_names[set_type_key]}.")

    # --- FINÁLNE OVERENIE A ŠTATISTIKY ---
    print("\n--- FINÁLNE ŠTATISTIKY A OVERENIE SAD ---")
    print(f"\n1. Celkový počet unikátnych zdrojových súborov: {total_source_files_count}")
    
    print(f"\n2. Počty súborov/párov v jednotlivých sadách:")
    print(f"   Zdrojové pre dynamický tréning: {len(source_files_for_dynamic_train_final)} súborov")
    print(f"   Statický referenčný tréningový: {final_set_counts['static_ref_train']} párov")
    print(f"   Statický validačný: {final_set_counts['static_valid']} párov")
    print(f"   Statický testovací: {final_set_counts['static_test']} párov")

    print("\n3. Kontrola konzistencie ID v statických sadách:")
    for split_key in ["static_ref_train", "static_valid", "static_test"]:
        set_name = dataset_names[split_key]
        images_dir = os.path.join(output_base_dir, set_name, "images")
        labels_dir = os.path.join(output_base_dir, set_name, "labels")
        
        image_ids_in_set = set()
        label_ids_in_set = set()
        
        if os.path.isdir(images_dir):
            for f_name in os.listdir(images_dir):
                match = re.search(r"wrappedbg_(\d{5})\.tiff", f_name)
                if match: image_ids_in_set.add(match.group(1))
        if os.path.isdir(labels_dir):
            for f_name in os.listdir(labels_dir):
                match = re.search(r"unwrapped_(\d{5})\.tiff", f_name)
                if match: label_ids_in_set.add(match.group(1))
        
        if image_ids_in_set == label_ids_in_set and len(image_ids_in_set) == final_set_counts[split_key]:
            print(f"   Sada '{set_name}': OK ({len(image_ids_in_set)} unikátnych a kompatibilných párov).")
        else:
            print(f"   CHYBA alebo UPOZORNENIE v sade '{set_name}':")
            print(f"     Nájdených ID v images: {len(image_ids_in_set)}, v labels: {len(label_ids_in_set)}, očakávaný počet párov: {final_set_counts[split_key]}")
            if image_ids_in_set != label_ids_in_set:
                 if image_ids_in_set - label_ids_in_set: print(f"       ID iba v images: {sorted(list(image_ids_in_set - label_ids_in_set))[:5]}...") # Len prvých 5
                 if label_ids_in_set - image_ids_in_set: print(f"       ID iba v labels: {sorted(list(label_ids_in_set - image_ids_in_set))[:5]}...") # Len prvých 5


    print("\n4. Zastúpenie kanálov v jednotlivých finálnych sadách (na základe zdrojových súborov):")
    # Zastúpenie kanálov pre dynamický tréning
    print(f"   Sada: {dataset_names['dynamic_train_source']}")
    channel_counts_dyn_train = defaultdict(int)
    for _, ch_name in source_files_for_dynamic_train_final: channel_counts_dyn_train[ch_name] += 1
    for ch_name in channel_names_ordered: print(f"     Kanál {ch_name}: {channel_counts_dyn_train.get(ch_name, 0)} súborov")

    # Zastúpenie kanálov pre statické sady (už vypočítané počas generovania)
    for set_key in ["static_ref_train", "static_valid", "static_test"]:
        print(f"   Sada: {dataset_names[set_key]}")
        for ch_name in channel_names_ordered:
            print(f"     Kanál {ch_name}: {final_channel_distribution[set_key].get(ch_name, 0)} párov")


    print("\nSpracovanie a rozdelenie datasetu dokončené.")


if __name__ == "__main__":
    base_data_path_config = "." 
    source_channel_directories_config = {
        "ch1": os.path.join(base_data_path_config, "collected_chanels_cleaned/ch1_filtered_default_exclusions"),
        "ch2": os.path.join(base_data_path_config, "collected_chanels_cleaned/ch2_filtered_default_exclusions"),
        "ch3": os.path.join(base_data_path_config, "collected_chanels_cleaned/ch3_filtered_default_exclusions"),
        "ch4": os.path.join(base_data_path_config, "collected_chanels_cleaned/ch4_filtered_default_exclusions")
    }
    
    output_main_dir_config = "split_dataset_tiff_for_dynamic_v_stratified_final"

    param_ranges_config_original = {
        "n_strips_param": (7, 8), 
        "original_image_influence_param": (0.3, 0.5),
        "phase_noise_std_param": (0.024, 0.039),
        "smooth_original_image_sigma_param": (0.2, 0.5),
        "poly_scale_param": (0.02, 0.1), 
        "CURVATURE_AMPLITUDE_param": (1.4, 2.0), 
        "background_offset_d_param": (-24.8, -6.8), 
        "tilt_angle_deg_param": (-5.0, 17.0) 
    }
    amplify_ab_fixed_config = 1.0

    print(f"Spúšťam skript na prípravu datasetov pre dynamický tréning (stratifikované delenie)...")
    print(f"Zdrojové kanály budú brané z: {source_channel_directories_config}")
    print(f"Cieľový výstupný priečinok: {output_main_dir_config}")
    
    create_final_datasets_stratified( 
        source_channel_dirs=source_channel_directories_config,
        output_base_dir=output_main_dir_config,
        simulation_param_ranges_config=param_ranges_config_original,
        amplify_ab_config_value=amplify_ab_fixed_config,
        ref_train_percentage_of_total=0.1,
        train_ratio_of_remaining=0.7,
        valid_ratio_of_remaining=0.15,
        random_seed=42
    )
    
    print(f"\nSkript na prípravu datasetov dokončil svoju prácu. Skontrolujte adresár '{output_main_dir_config}'.")
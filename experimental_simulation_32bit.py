import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.filters import gaussian
from skimage.morphology import disk
# from skimage.filters import gaussian as skimage_gaussian_placeholder # Nepotrebujeme
import os
import glob
import time
import tifffile # Importujeme tifffile pre ukladanie float32 TIFFov

# --- Pomocná funkcia ---
def get_random_or_fixed(param_value, is_integer=False, allow_float_for_int=False):
    if isinstance(param_value, (list, tuple)) and len(param_value) == 2:
        min_val, max_val = param_value
        if is_integer:
            if not allow_float_for_int:
                return np.random.randint(int(min_val), int(max_val) + 1)
            else:
                return int(np.random.randint(int(round(min_val)), int(round(max_val)) + 1))
        else:
            return np.random.uniform(min_val, max_val)
    return param_value

# --- Zabalenie fázy ---
def wrap_phase(img):
    return (img + np.pi) % (2 * np.pi) - np.pi

# --- Generovanie pozadia s možnosťou naklonenia ---
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


# --- Hlavná simulačná funkcia pre generovanie a ukladanie jednej dvojice ako float32 ---
def generate_and_save_simulation_pair_float32(
    input_filename,
    output_dir_unwrapped,
    output_dir_wrapped,
    # Parametre, ktoré môžu byť zadané ako rozsahy (min, max)
    n_strips_param=(6, 7), 
    original_image_influence_param=(0.3, 0.45), # Mierne zvýšená dolná hranica pre viditeľnosť buniek
    phase_noise_std_param=(0.02, 0.045),
    smooth_original_image_sigma_param=(0.7, 1.8),
    poly_scale_param=(0.02, 0.1),
    CURVATURE_AMPLITUDE_param=(1.0, 3.0), 
    background_offset_d_param=(-20.0, 0.0),
    tilt_angle_deg_param=(-5.0, 17.0), 
    amplify_ab=1.0,
    verbose=True
):
    n_strips = get_random_or_fixed(n_strips_param, is_integer=True, allow_float_for_int=True)
    original_image_influence = get_random_or_fixed(original_image_influence_param)
    phase_noise_std = get_random_or_fixed(phase_noise_std_param)
    smooth_original_image_sigma = get_random_or_fixed(smooth_original_image_sigma_param)
    poly_scale = get_random_or_fixed(poly_scale_param)
    CURVATURE_AMPLITUDE = get_random_or_fixed(CURVATURE_AMPLITUDE_param)
    background_d_offset = get_random_or_fixed(background_offset_d_param)
    tilt_angle_deg = get_random_or_fixed(tilt_angle_deg_param)

    coeff_stats = [
        (0.0, 0.3 * CURVATURE_AMPLITUDE),
        (-4.0 * CURVATURE_AMPLITUDE, 0.3 * CURVATURE_AMPLITUDE),
        (+4.0 * CURVATURE_AMPLITUDE, 0.3 * CURVATURE_AMPLITUDE),
        (background_d_offset, 2.0)
    ]

    if verbose:
        print(f"  Parametre pre {os.path.basename(input_filename)}:")
        print(f"    n_strips: {n_strips}, obj_influence: {original_image_influence:.2f}, noise_std: {phase_noise_std:.3f}")
        print(f"    smooth_sigma: {smooth_original_image_sigma:.2f}, poly_scale: {poly_scale:.3f}, curv_amp: {CURVATURE_AMPLITUDE:.2f}")
        print(f"    d_off:{background_d_offset:.2f}, tilt_angle: {tilt_angle_deg:.2f} deg")

    try:
        img_raw = io.imread(input_filename).astype(np.float32) # Vstup je uint16, ale hneď konvertujeme na float pre výpočty
    except Exception as e:
        print(f"  CHYBA: Nepodarilo sa načítať {input_filename}: {e}. Tento obrázok bude preskočený.")
        return False

    img_min, img_max = img_raw.min(), img_raw.max()
    img_norm = (img_raw - img_min) / (img_max - img_min) if img_max > img_min else np.zeros_like(img_raw)

    if smooth_original_image_sigma > 0 and original_image_influence > 0:
        img_phase_obj = gaussian(img_norm, sigma=smooth_original_image_sigma, preserve_range=True)
    else:
        img_phase_obj = img_norm
    
    object_phase_contribution = img_phase_obj * (2 * np.pi)

    background, _ = generate_cubic_background(
        img_raw.shape, coeff_stats, scale=poly_scale,
        amplify_ab=amplify_ab, n_strips=n_strips,
        tilt_angle_deg=tilt_angle_deg
    )
    unwrapped_phase = (object_phase_contribution * original_image_influence) + \
                      (background * (1.0 - original_image_influence))

    if phase_noise_std > 0:
        unwrapped_phase += np.random.normal(0, phase_noise_std, size=img_raw.shape)
    
    wrapped_phase = wrap_phase(unwrapped_phase) # Toto je stále v radiánoch (-pi, pi)

    if verbose: 
        min_r, max_r = unwrapped_phase.min(), unwrapped_phase.max()
        print(f"    Unwrapped Phase (rad): Min={min_r:.2f}, Max={max_r:.2f}, Range={max_r - min_r:.2f}")
        min_w, max_w = wrapped_phase.min(), wrapped_phase.max()
        print(f"    Wrapped Phase (rad): Min={min_w:.2f}, Max={max_w:.2f}") # Malo by byť blízko -pi, pi

    # Ukladanie priamo ako float32
    unwrapped_to_save_f32 = unwrapped_phase.astype(np.float32)
    wrapped_to_save_f32 = wrapped_phase.astype(np.float32) # wrapped_phase je už float
    
    base_name_orig = os.path.splitext(os.path.basename(input_filename))[0]
    # Môžeme pridať sufix "_f32" k názvu, aby sme odlíšili od prípadných uint16 verzií
    output_filename_unwrapped = os.path.join(output_dir_unwrapped, f"{base_name_orig}_f32.tiff")
    output_filename_wrapped = os.path.join(output_dir_wrapped, f"{base_name_orig}_f32.tiff")

    try:
        tifffile.imwrite(output_filename_unwrapped, unwrapped_to_save_f32)
        if verbose: print(f"    Uložený (float32): {output_filename_unwrapped}")
        
        tifffile.imwrite(output_filename_wrapped, wrapped_to_save_f32)
        if verbose: print(f"    Uložený (float32): {output_filename_wrapped}")
        return True
    except Exception as e:
        print(f"  CHYBA pri ukladaní (tifffile) pre {base_name_orig}: {e}")
        return False


# --- Hlavná časť skriptu pre generovanie datasetu ---
if __name__ == '__main__':
    start_time = time.time()

    # --- Konfigurácia ---
    input_image_directory = "collected_chanels_cleaned/ch3_filtered_default_exclusions" # Cesta k adresáru s pôvodnými obrázkami
    output_dataset_base_dir = "simulated_dataset_ch3" # Nový názov adresára

    param_ranges = {
        "n_strips_param": (7, 8), 
        "original_image_influence_param": (0.3, 0.5), # Mierne zvýšená dolná aj horná hranica
        "phase_noise_std_param": (0.024, 0.039),
        "smooth_original_image_sigma_param": (0.2, 0.5), # Mierne znížená horná pre ostrejšie bunky
        "poly_scale_param": (0.02, 0.1),
        "CURVATURE_AMPLITUDE_param": (1.4, 2.0), 
        "background_offset_d_param": (-24.8, -6.8), 
        "tilt_angle_deg_param": (-5.0, 17.0) 
    }
    # --------------------

    output_unwrapped_dir = os.path.join(output_dataset_base_dir, "unwrapped")
    output_wrapped_dir = os.path.join(output_dataset_base_dir, "wrappedbg")
    os.makedirs(output_unwrapped_dir, exist_ok=True)
    os.makedirs(output_wrapped_dir, exist_ok=True)

    original_image_files = glob.glob(os.path.join(input_image_directory, "**", "*.tiff"), recursive=True)
    
    if not original_image_files:
        print(f"CHYBA: Nenašli sa žiadne .tiff súbory v: {input_image_directory}")
        exit()
    
    print(f"Nájdených {len(original_image_files)} pôvodných obrázkov.")
    print(f"Pre každý pôvodný obrázok sa vygeneruje jedna simulovaná dvojica (ukladaná ako float32).")

    simulations_completed = 0
    simulations_failed = 0

    for i, orig_img_path in enumerate(original_image_files):
        print(f"\nSpracovávam pôvodný obrázok ({i+1}/{len(original_image_files)}): {orig_img_path}")
        
        success = generate_and_save_simulation_pair_float32( # Voláme novú funkciu
            input_filename=orig_img_path,
            output_dir_unwrapped=output_unwrapped_dir,
            output_dir_wrapped=output_wrapped_dir,
            n_strips_param=param_ranges["n_strips_param"],
            original_image_influence_param=param_ranges["original_image_influence_param"],
            phase_noise_std_param=param_ranges["phase_noise_std_param"],
            smooth_original_image_sigma_param=param_ranges["smooth_original_image_sigma_param"],
            poly_scale_param=param_ranges["poly_scale_param"],
            CURVATURE_AMPLITUDE_param=param_ranges["CURVATURE_AMPLITUDE_param"],
            background_offset_d_param=param_ranges["background_offset_d_param"],
            tilt_angle_deg_param=param_ranges["tilt_angle_deg_param"],
            verbose=True
        )
        if success:
            simulations_completed += 1
        else:
            simulations_failed +=1
            
    end_time = time.time()
    print(f"\nGenerovanie datasetu dokončené.")
    print(f"Počet úspešne vygenerovaných a uložených dvojíc: {simulations_completed}")
    if simulations_failed > 0:
        print(f"Počet neúspešných pokusov (chyba pri načítaní/ukladaní): {simulations_failed}")
    print(f"Trvanie: {end_time - start_time:.2f} sekúnd.")
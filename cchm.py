import os
import glob
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import tifffile
import segmentation_models_pytorch as smp
from tqdm import tqdm
from skimage.restoration import unwrap_phase as skimage_unwrap_phase
import re

# --- Globálne Konštanty ---
TARGET_IMG_SIZE = (512, 512)
KMAX_FALLBACK_DWC = 6
DEVICE_TO_USE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Pomocné Funkcie (zdieľané z GRAND_finale.py) ---
def _ensure_shape_global(img_numpy, target_shape, data_name="Image", dtype=np.float32):
    if img_numpy is None:
        print(f"CHYBA: Vstup pre _ensure_shape_global ({data_name}) je None.")
        return None
    try:
        img_numpy = np.array(img_numpy, dtype=dtype)
    except Exception as e:
        print(f"CHYBA pri konverzii na NumPy pole v _ensure_shape_global ({data_name}): {e}")
        return None
    if img_numpy.ndim == 0:
        print(f"CHYBA: Načítaný obrázok {data_name} je skalár.")
        return None
    current_shape = img_numpy.shape[-2:]
    if current_shape != target_shape:
        original_shape_for_debug = img_numpy.shape
        h, w = current_shape; target_h, target_w = target_shape
        pad_h = max(0, target_h - h); pad_w = max(0, target_w - w)
        if pad_h > 0 or pad_w > 0:
            pad_top = pad_h // 2; pad_bottom = pad_h - (pad_h // 2)
            pad_left = pad_w // 2; pad_right = pad_w - (pad_w // 2)
            if img_numpy.ndim == 2:
                padding_dims = ((pad_top, pad_bottom), (pad_left, pad_right))
            elif img_numpy.ndim == 3 and img_numpy.shape[0] == 1 :
                padding_dims = ((0,0), (pad_top, pad_bottom), (pad_left, pad_right))
            else:
                print(f"Nepodporovaný tvar pre padding ({data_name}): {original_shape_for_debug}.")
                return None
            img_numpy = np.pad(img_numpy, padding_dims, mode='reflect')
        h, w = img_numpy.shape[-2:]
        if h > target_h or w > target_w:
            start_h = (h - target_h) // 2; start_w = (w - target_w) // 2
            if img_numpy.ndim == 2: img_numpy = img_numpy[start_h:start_h+target_h, start_w:start_w+target_w]
            elif img_numpy.ndim == 3 and img_numpy.shape[0] == 1: img_numpy = img_numpy[:,start_h:start_h+target_h, start_w:start_w+target_w]
            else:
                print(f"Nepodporovaný tvar pre cropping ({data_name}): {original_shape_for_debug}.")
                return None
        if img_numpy.shape[-2:] != target_shape:
             print(f"VAROVANIE: {data_name} mal tvar {original_shape_for_debug}, po úprave na {target_shape} má {img_numpy.shape[-2:]}.")
             return None
    return img_numpy

def denormalize_target_z_score_global(data_norm, original_mean, original_std):
    if abs(original_std) < 1e-7: return torch.full_like(data_norm, original_mean)
    return data_norm * original_std + original_mean

# --- Dataset a funkcie pre dRG (len inferencia) ---
class CustomDataset_dRG_Inference(Dataset):
    def __init__(self, image_files_wrapped, norm_stats_input, target_img_size):
        self.image_files_wrapped = image_files_wrapped
        if not self.image_files_wrapped:
            raise FileNotFoundError(f"Nenašli sa žiadne vstupné wrappedbg_*.tiff súbory.")
        self.input_min, self.input_max = norm_stats_input
        self.target_img_size = target_img_size

    def _normalize_input_minmax(self, data, min_val, max_val):
        if max_val == min_val: return torch.zeros_like(data)
        return 2.0 * (data - min_val) / (max_val - min_val) - 1.0
        
    def __len__(self): return len(self.image_files_wrapped)

    def __getitem__(self, index):
        wrapped_img_path = self.image_files_wrapped[index]
        base_id_name = os.path.basename(wrapped_img_path).replace('wrappedbg_', '').replace('.tiff', '')
        try:
            wrapped_phase_original_np = tifffile.imread(wrapped_img_path)
        except Exception as e:
            print(f"CHYBA (dRG Dataset __getitem__): {e} pri {wrapped_img_path}"); return None, None
        
        wrapped_for_input = _ensure_shape_global(wrapped_phase_original_np.copy(), self.target_img_size, f"dRG_Input ({os.path.basename(wrapped_img_path)})")
        if wrapped_for_input is None: return None, None

        input_tensor = torch.from_numpy(wrapped_for_input.astype(np.float32))
        input_tensor_norm = self._normalize_input_minmax(input_tensor, self.input_min, self.input_max).unsqueeze(0)
        
        return input_tensor_norm, base_id_name

def collate_fn_dRG_inference(batch):
    batch = list(filter(lambda x: x is not None and x[0] is not None and x[1] is not None, batch))
    if not batch: return None, None
    inputs = torch.utils.data.dataloader.default_collate([item[0] for item in batch])
    ids = [item[1] for item in batch]
    return inputs, ids

def run_dRG_inference(config_path, weights_path, input_image_files, output_dir_drg, device_str):
    config = {};
    try:
        with open(config_path, 'r') as f:
            for line in f:
                if ":" in line: key, value = line.split(":", 1); config[key.strip()] = value.strip()
    except Exception as e:
        print(f"CHYBA pri čítaní config súboru pre dRG ({config_path}): {e}"); return

    encoder_name = config.get("Encoder Name", "resnet34")
    target_mean = float(config.get("Target Norm (Z-score) Mean", 0.0))
    target_std = float(config.get("Target Norm (Z-score) Std", 1.0))
    input_min = float(config.get("Input Norm (MinMax) Min", -np.pi))
    input_max = float(config.get("Input Norm (MinMax) Max", np.pi))

    dataset = CustomDataset_dRG_Inference(input_image_files, (input_min, input_max), TARGET_IMG_SIZE)
    if len(dataset) == 0: print("dRG: Vstupný dataset prázdny."); return
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn_dRG_inference)
    
    device = torch.device(device_str); print(f"dRG používa: {device}")
    model = smp.Unet(encoder_name=encoder_name, encoder_weights=None, in_channels=1, classes=1, activation=None).to(device)
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    except Exception as e: print(f"CHYBA nacitania vah dRG ({weights_path}): {e}"); return
    model.eval()

    print("Spracovanie dRG modelu (inferencia)...")
    with torch.no_grad():
        for data_batch in tqdm(loader, desc="dRG Inference"):
            if data_batch is None or data_batch[0] is None: continue
            input_norm_tensor, base_id_names = data_batch
            
            if input_norm_tensor is None: continue
            base_id_name = base_id_names[0] # Batch size is 1

            pred_norm_tensor = model(input_norm_tensor.to(device))
            psi_M_denorm_torch = denormalize_target_z_score_global(
                pred_norm_tensor.squeeze(0), 
                target_mean, 
                target_std
            )
            unwrapped_pred_np_orig = psi_M_denorm_torch.cpu().numpy().squeeze()
            
            unwrapped_pred_np = _ensure_shape_global(unwrapped_pred_np_orig, TARGET_IMG_SIZE, f"dRG_Output ({base_id_name})")
            if unwrapped_pred_np is None:
                print(f"CHYBA: _ensure_shape_global zlyhalo pre dRG výstup ({base_id_name}). Preskakujem ukladanie.")
                continue

            output_path = os.path.join(output_dir_drg, f"unwrapped_{base_id_name}.tiff")
            try:
                tifffile.imwrite(output_path, unwrapped_pred_np.astype(np.float32))
            except Exception as e:
                print(f"CHYBA pri ukladaní dRG výsledku pre {base_id_name}: {e}")

# --- dWC Metóda: Dataset, Inferencia ---
class CustomDataset_dWC_Inference(Dataset):
    def __init__(self, image_files_wrapped, norm_stats_input, target_img_size):
        self.image_files_wrapped = image_files_wrapped
        if not self.image_files_wrapped:
            raise FileNotFoundError(f"Nenašli sa žiadne vstupné wrappedbg_*.tiff súbory.")
        self.input_min_g, self.input_max_g = norm_stats_input
        self.target_img_size = target_img_size

    def _normalize_input_minmax(self, data, min_val, max_val):
        # Vráti numpy pole pre normalizovaný vstup
        if max_val == min_val: return np.zeros_like(data)
        return 2.0 * (data - min_val) / (max_val - min_val) - 1.0

    def __len__(self): return len(self.image_files_wrapped)

    def __getitem__(self, index):
        wrapped_img_path = self.image_files_wrapped[index]
        base_id_name = os.path.basename(wrapped_img_path).replace('wrappedbg_', '').replace('.tiff', '')
        try:
            wrapped_orig_np = tifffile.imread(wrapped_img_path)
        except Exception as e:
            print(f"CHYBA (dWC Dataset __getitem__): {e} pri {wrapped_img_path}"); return None, None, None
        
        wrapped_for_input_np = _ensure_shape_global(wrapped_orig_np.copy(), self.target_img_size, f"dWC_Input ({os.path.basename(wrapped_img_path)})")
        if wrapped_for_input_np is None: return None, None, None

        wrapped_input_norm_np = self._normalize_input_minmax(wrapped_for_input_np, self.input_min_g, self.input_max_g)
        wrapped_input_norm_tensor = torch.from_numpy(wrapped_input_norm_np.astype(np.float32)).unsqueeze(0)
        
        return wrapped_input_norm_tensor, wrapped_for_input_np.astype(np.float32), base_id_name

def collate_fn_dWC_inference(batch):
    batch = list(filter(lambda x: x is not None and all(item is not None for item in x), batch))
    if not batch: return None, None, None
    norm_tensors = torch.utils.data.dataloader.default_collate([item[0] for item in batch])
    orig_numpy_list = [item[1] for item in batch] # Zostane ako zoznam numpy polí
    ids = [item[2] for item in batch]
    return norm_tensors, orig_numpy_list, ids


def run_dWC_inference(config_path_clf, weights_path_clf, input_image_files, output_dir_dwc, device_str):
    config = {}; 
    try:
        with open(config_path_clf, 'r') as f:
            for line in f:
                if ":" in line: key, value = line.split(":", 1); config[key.strip()] = value.strip()
    except Exception as e:
        print(f"CHYBA pri čítaní config súboru pre dWC ({config_path_clf}): {e}"); return

    encoder_name = config.get("Encoder Name", "resnet34")
    k_max_val = int(config.get("K_MAX", KMAX_FALLBACK_DWC))
    num_classes_effective = 2 * k_max_val + 1
    
    input_norm_str = config.get("Input Normalization (Global MinMax for Wrapped)", f"Min: {-np.pi:.4f}, Max: {np.pi:.4f}")
    global_input_min, global_input_max = -np.pi, np.pi 
    try:
        min_str_part = re.search(r"Min:\s*([-\d.]+)", input_norm_str)
        max_str_part = re.search(r"Max:\s*([-\d.]+)", input_norm_str)
        if min_str_part and max_str_part:
            global_input_min = float(min_str_part.group(1))
            global_input_max = float(max_str_part.group(1))
    except Exception as e_parse:
        print(f"CHYBA parsovania Min/Max pre dWC: {e_parse}. Používam default.")

    dataset = CustomDataset_dWC_Inference(input_image_files, (global_input_min, global_input_max), TARGET_IMG_SIZE)
    if len(dataset) == 0: print("dWC: Vstupný dataset prázdny."); return
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn_dWC_inference)

    device = torch.device(device_str); print(f"dWC používa: {device}")
    model = smp.Unet(encoder_name=encoder_name, encoder_weights=None, in_channels=1, classes=num_classes_effective, activation=None).to(device)
    try:
        model.load_state_dict(torch.load(weights_path_clf, map_location=device, weights_only=True))
    except Exception as e: print(f"CHYBA nacitania vah dWC ({weights_path_clf}): {e}"); return
    model.eval()

    print("Spracovanie dWC modelu (inferencia)...")
    with torch.no_grad():
        for data_batch in tqdm(loader, desc="dWC Inference"):
            if data_batch is None or data_batch[0] is None: continue
            input_norm_tensor, wrapped_orig_np_list, base_id_names = data_batch
            
            if input_norm_tensor is None : continue
            
            wrapped_orig_np = wrapped_orig_np_list[0] # Batch size is 1
            base_id_name = base_id_names[0]       # Batch size is 1

            logits = model(input_norm_tensor.to(device))
            pred_classes = torch.argmax(logits.squeeze(0), dim=0).cpu() 
            k_pred_values = pred_classes.float() - k_max_val
            
            reconstructed_phase_orig = wrapped_orig_np + (2 * np.pi) * k_pred_values.numpy()
            
            reconstructed_phase_np = _ensure_shape_global(reconstructed_phase_orig, TARGET_IMG_SIZE, f"dWC_Output ({base_id_name})")
            if reconstructed_phase_np is None:
                print(f"CHYBA: _ensure_shape_global zlyhalo pre dWC výstup ({base_id_name}). Preskakujem ukladanie.")
                continue
            
            output_path = os.path.join(output_dir_dwc, f"unwrapped_{base_id_name}.tiff")
            try:
                tifffile.imwrite(output_path, reconstructed_phase_np.astype(np.float32))
            except Exception as e:
                print(f"CHYBA pri ukladaní dWC výsledku pre {base_id_name}: {e}")


# --- L2 Metóda (Skimage) ---
def run_L2_inference(input_image_files, output_dir_l2):
    print("Spracovanie L2 (skimage.restoration.unwrap_phase)...")
    if not input_image_files:
        print("L2: Nenašli sa žiadne vstupné súbory.")
        return

    for img_path in tqdm(input_image_files, desc="L2 Inference"):
        base_id_name = os.path.basename(img_path).replace('wrappedbg_', '').replace('.tiff', '')
        try:
            wrapped_img_orig = tifffile.imread(img_path)
            wrapped_img = _ensure_shape_global(wrapped_img_orig, TARGET_IMG_SIZE, f"L2_Input ({os.path.basename(img_path)})")
            if wrapped_img is None: 
                print(f"CHYBA: _ensure_shape_global zlyhalo pre L2 vstup ({base_id_name}). Preskakujem.")
                continue

            unwrapped_l2_orig = skimage_unwrap_phase(wrapped_img)
            unwrapped_l2 = _ensure_shape_global(unwrapped_l2_orig, TARGET_IMG_SIZE, f"L2_Output ({base_id_name})")
            if unwrapped_l2 is None :
                print(f"CHYBA: _ensure_shape_global zlyhalo pre L2 výstup ({base_id_name}). Preskakujem ukladanie.")
                continue
            
            output_path = os.path.join(output_dir_l2, f"unwrapped_{base_id_name}.tiff")
            tifffile.imwrite(output_path, unwrapped_l2.astype(np.float32))

        except Exception as e:
            print(f"Chyba pri L2 pre {img_path}: {e}")

# --- Hlavný skript ---
if __name__ == '__main__':
    # --- KONFIGURÁCIA CIEST ---
    # Upravte tieto cesty podľa vašej štruktúry
    CONFIG_PATH_DRG = r"C:\Users\viera\Desktop\q_tiff\TRENOVANIE_bakalarka_simul\optimalizacia_hype\trenovanie_5\config_R34imgnet_direct_MAE_GDL0p3_AugMedium_LR5em04_WD1em04_Ep120_ESp30_Tmax120_EtaMin1em07_bs8_bs8.txt" 
    WEIGHTS_PATH_DRG = r"C:\Users\viera\Desktop\q_tiff\TRENOVANIE_bakalarka_simul\optimalizacia_hype\trenovanie_5\best_weights_R34imgnet_direct_MAE_GDL0p3_AugMedium_LR5em04_WD1em04_Ep120_ESp30_Tmax120_EtaMin1em07_bs8_bs8.pth"
    
    CONFIG_PATH_DWC = r"C:\Users\viera\Desktop\q_tiff\TRENOVANIE_bakalarka_simul\classification\experiment4_hyper\config_clf_R34imgnet_Kmax6_AugMed_LR1em03_WD1em04_Ep120_Tmax120_EtaMin1em07_EdgeW5.0_bs8.txt"
    WEIGHTS_PATH_DWC = r"C:\Users\viera\Desktop\q_tiff\TRENOVANIE_bakalarka_simul\classification\experiment4_hyper\best_weights_clf_R34imgnet_Kmax6_AugMed_LR1em03_WD1em04_Ep120_Tmax120_EtaMin1em07_EdgeW5.0_bs8.pth"
    
    # Cesta k adresáru, ktorý obsahuje podadresár 'images' s 'wrappedbg_*.tiff' súbormi
    INPUT_DATA_ROOT = r'C:\Users\viera\Desktop\cchm_ttet' 
    
    # Hlavný výstupný adresár pre tento skript
    MAIN_OUTPUT_DIR = r"C:\Users\viera\Desktop\GRAAFY\CCHM_RESULTS" 
    # --------------------------

    start_time_overall = time.time()

    # Vytvorenie výstupných adresárov
    output_dir_drg = os.path.join(MAIN_OUTPUT_DIR, "dRG_unwrapped")
    output_dir_dwc = os.path.join(MAIN_OUTPUT_DIR, "dWC_unwrapped")
    output_dir_l2 = os.path.join(MAIN_OUTPUT_DIR, "L2_unwrapped")

    os.makedirs(output_dir_drg, exist_ok=True)
    os.makedirs(output_dir_dwc, exist_ok=True)
    os.makedirs(output_dir_l2, exist_ok=True)

    # Získanie zoznamu vstupných obrázkov
    input_images_path = os.path.join(INPUT_DATA_ROOT, 'images')
    if not os.path.isdir(input_images_path):
        print(f"CHYBA: Adresár so vstupnými obrázkami '{input_images_path}' neexistuje.")
        exit()
    
    input_image_files = sorted(glob.glob(os.path.join(input_images_path, "wrappedbg_*.tiff")))
    if not input_image_files:
        print(f"CHYBA: Nenašli sa žiadne 'wrappedbg_*.tiff' súbory v '{input_images_path}'.")
        exit()

    print(f"Nájdených {len(input_image_files)} vstupných obrázkov.")

    # --- Spustenie inferencie pre dRG ---
    print("\n--- Spúšťam dRG inferenciu ---")
    if os.path.exists(CONFIG_PATH_DRG) and os.path.exists(WEIGHTS_PATH_DRG):
        run_dRG_inference(CONFIG_PATH_DRG, WEIGHTS_PATH_DRG, input_image_files, output_dir_drg, DEVICE_TO_USE)
    else:
        print(f"CHYBA: Chýbajú konfiguračné súbory alebo váhy pre dRG. Preskakujem.")

    # --- Spustenie inferencie pre dWC ---
    print("\n--- Spúšťam dWC inferenciu ---")
    if os.path.exists(CONFIG_PATH_DWC) and os.path.exists(WEIGHTS_PATH_DWC):
        run_dWC_inference(CONFIG_PATH_DWC, WEIGHTS_PATH_DWC, input_image_files, output_dir_dwc, DEVICE_TO_USE)
    else:
        print(f"CHYBA: Chýbajú konfiguračné súbory alebo váhy pre dWC. Preskakujem.")

    # --- Spustenie inferencie pre L2 ---
    print("\n--- Spúšťam L2 (skimage) inferenciu ---")
    run_L2_inference(input_image_files, output_dir_l2)
    
    total_script_time_overall = time.time() - start_time_overall
    print(f"\nCelkové spracovanie dokončené za {time.strftime('%H:%M:%S', time.gmtime(total_script_time_overall))}.")
    print(f"Výsledky uložené v adresári: {MAIN_OUTPUT_DIR}")

    print("Hotovo.")  
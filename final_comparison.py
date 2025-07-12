import os
import glob
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import tifffile
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import MaxNLocator
import segmentation_models_pytorch as smp
from tqdm import tqdm
from skimage.restoration import unwrap_phase as skimage_unwrap_phase
from skimage.metrics import peak_signal_noise_ratio as ski_psnr
from skimage.metrics import structural_similarity as ski_ssim
import re

# --- Globálne Konštanty ---
TARGET_IMG_SIZE = (512, 512)
KMAX_FALLBACK_DWC = 6

# --- Pomocné Funkcie (zdieľané) ---
def calculate_mae(image1, image2):
    if image1.shape != image2.shape: return np.nan
    return np.mean(np.abs(image1 - image2))

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
                print(f"Nepodporovaný tvar pre padding ({data_name}): {original_shape_for_debug}. Path: {getattr(img_numpy, 'filename', 'N/A') if hasattr(img_numpy, 'filename') else 'N/A'}")
                return None
            img_numpy = np.pad(img_numpy, padding_dims, mode='reflect')
        h, w = img_numpy.shape[-2:]
        if h > target_h or w > target_w:
            start_h = (h - target_h) // 2; start_w = (w - target_w) // 2
            if img_numpy.ndim == 2: img_numpy = img_numpy[start_h:start_h+target_h, start_w:start_w+target_w]
            elif img_numpy.ndim == 3 and img_numpy.shape[0] == 1: img_numpy = img_numpy[:,start_h:start_h+target_h, start_w:start_w+target_w]
            else:
                print(f"Nepodporovaný tvar pre cropping ({data_name}): {original_shape_for_debug}. Path: {getattr(img_numpy, 'filename', 'N/A') if hasattr(img_numpy, 'filename') else 'N/A'}")
                return None
        if img_numpy.shape[-2:] != target_shape:
             print(f"VAROVANIE: {data_name} mal tvar {original_shape_for_debug}, po úprave na {target_shape} má {img_numpy.shape[-2:]}. Path: {getattr(img_numpy, 'filename', 'N/A') if hasattr(img_numpy, 'filename') else 'N/A'}")
             return None
    return img_numpy

def calculate_psnr_ssim_global(gt_img_numpy, pred_img_numpy):
    if gt_img_numpy is None or pred_img_numpy is None: return np.nan, np.nan
    gt_img_numpy = gt_img_numpy.squeeze().astype(np.float32)
    pred_img_numpy = pred_img_numpy.squeeze().astype(np.float32)
    if gt_img_numpy.shape != pred_img_numpy.shape: return np.nan, np.nan
    data_range = gt_img_numpy.max() - gt_img_numpy.min()
    current_psnr, current_ssim = np.nan, np.nan
    if data_range < 1e-6:
        current_psnr = float('inf') if np.allclose(gt_img_numpy, pred_img_numpy) else 0.0
    else:
        try: current_psnr = ski_psnr(gt_img_numpy, pred_img_numpy, data_range=data_range)
        except Exception: pass # PSNR môže zlyhať pri zlých vstupoch
    min_dim = min(gt_img_numpy.shape[-2:])
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    if win_size >= 3:
        try: current_ssim = ski_ssim(gt_img_numpy, pred_img_numpy, data_range=data_range, channel_axis=None, win_size=win_size, gaussian_weights=True, use_sample_covariance=False)
        except Exception: pass
    return current_psnr, current_ssim

def denormalize_target_z_score_global(data_norm, original_mean, original_std):
    if abs(original_std) < 1e-7: return torch.full_like(data_norm, original_mean)
    return data_norm * original_std + original_mean

# --- Dataset a funkcie pre dRG (BEZ KONGRUENCIE V TEJTO EVALUÁCII) ---
class CustomDataset_dRG_Eval(Dataset):
    def __init__(self, path_to_data, norm_stats_input, norm_stats_target, target_img_size):
        self.path = path_to_data
        # Párovanie wrapped a unwrapped súborov
        self.image_files_wrapped = sorted(glob.glob(os.path.join(self.path, 'images', "wrappedbg_*.tiff")))
        self.label_files_unwrapped_gt = [
            os.path.join(os.path.dirname(os.path.dirname(fp)), 'labels', os.path.basename(fp).replace('wrappedbg_', 'unwrapped_'))
            for fp in self.image_files_wrapped
        ]
        
        # Overenie a filtrovanie párov
        valid_pairs = []
        for i, (img_fp, lbl_fp) in enumerate(zip(self.image_files_wrapped, self.label_files_unwrapped_gt)):
            if os.path.exists(img_fp) and os.path.exists(lbl_fp):
                valid_pairs.append((img_fp, lbl_fp))
            else:
                print(f"VAROVANIE (dRG Dataset): Chýbajúci súbor pre pár: {img_fp} alebo {lbl_fp}. Preskakujem.")
        
        self.image_files_wrapped = [pair[0] for pair in valid_pairs]
        self.label_files_unwrapped_gt = [pair[1] for pair in valid_pairs]

        if not self.image_files_wrapped:
            raise FileNotFoundError(f"Nenašli sa žiadne platné páry vstupných/výstupných obrazov v {path_to_data}")

        self.input_min, self.input_max = norm_stats_input
        self.target_mean, self.target_std = norm_stats_target
        self.target_img_size = target_img_size
        self.current_img_path_for_debug = "N/A"


    def _normalize_input_minmax(self, data, min_val, max_val):
        if max_val == min_val: return torch.zeros_like(data)
        return 2.0 * (data - min_val) / (max_val - min_val) - 1.0
        
    def __len__(self): return len(self.image_files_wrapped)

    def __getitem__(self, index):
        wrapped_img_path = self.image_files_wrapped[index]
        unwrapped_gt_path = self.label_files_unwrapped_gt[index]
        self.current_img_path_for_debug = wrapped_img_path
        try:
            wrapped_phase_original_np = tifffile.imread(wrapped_img_path)
            unwrapped_gt_original_np = tifffile.imread(unwrapped_gt_path)
        except Exception as e:
            print(f"CHYBA (dRG Dataset __getitem__): {e} pri {wrapped_img_path}"); return None, None
        
        wrapped_for_input = _ensure_shape_global(wrapped_phase_original_np.copy(), self.target_img_size, f"dRG_Input ({os.path.basename(wrapped_img_path)})")
        gt_for_eval = _ensure_shape_global(unwrapped_gt_original_np.copy(), self.target_img_size, f"dRG_GT ({os.path.basename(unwrapped_gt_path)})")

        if wrapped_for_input is None or gt_for_eval is None:
             return None, None # Vraciame len 2 hodnoty

        input_tensor = torch.from_numpy(wrapped_for_input.astype(np.float32))
        input_tensor_norm = self._normalize_input_minmax(input_tensor, self.input_min, self.input_max).unsqueeze(0)
        
        return input_tensor_norm, gt_for_eval.astype(np.float32) # Vstup pre model a GT

def collate_fn_dRG_eval(batch):
    batch = list(filter(lambda x: x is not None and all(item is not None for item in x), batch))
    if not batch: return None, None
    try: return torch.utils.data.dataloader.default_collate(batch)
    except Exception as e: print(f"CHYBA v collate_fn_dRG_eval: {e}"); return None, None

def evaluate_dRG_method(config_path, weights_path, test_dataset_path, device_str):
    config = {};
    with open(config_path, 'r') as f:
        for line in f:
            if ":" in line: key, value = line.split(":", 1); config[key.strip()] = value.strip()
    encoder_name = config.get("Encoder Name", "resnet34")
    target_mean = float(config.get("Target Norm (Z-score) Mean", 0.0))
    target_std = float(config.get("Target Norm (Z-score) Std", 1.0))
    input_min = float(config.get("Input Norm (MinMax) Min", -np.pi))
    input_max = float(config.get("Input Norm (MinMax) Max", np.pi))

    dataset = CustomDataset_dRG_Eval(test_dataset_path, (input_min, input_max), (target_mean, target_std), TARGET_IMG_SIZE)
    if len(dataset) == 0: print("dRG: Testovací dataset prázdny."); return [], [], [], []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn_dRG_eval)
    
    device = torch.device(device_str); print(f"dRG používa: {device}")
    model = smp.Unet(encoder_name=encoder_name, encoder_weights=None, in_channels=1, classes=1, activation=None).to(device)
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    except Exception as e: print(f"CHYBA nacitania vah dRG: {e}"); return [],[],[],[]
    model.eval()

    pixel_errs, img_maes, img_psnrs, img_ssims = [], [], [], []
    print("Spracovanie dRG modelu (BEZ operácie kongruencie)...") # ZMENA VÝPISU
    with torch.no_grad():
        for data_batch in tqdm(loader, desc="dRG Eval (no congruency)"):
            if data_batch is None or data_batch[0] is None: continue
            input_norm_tensor, gt_np_batch = data_batch # Už len 2 položky
            
            if input_norm_tensor is None: continue

            pred_norm_tensor = model(input_norm_tensor.to(device))
            # Denormalizujeme priamu predikciu siete
            psi_M_denorm_np = denormalize_target_z_score_global(
                pred_norm_tensor.squeeze(0), 
                target_mean, 
                target_std
            ).cpu().numpy().squeeze()
            
            gt_np = gt_np_batch.cpu().numpy().squeeze()

            # Chyba sa počíta z priamej predikcie psi_M
            errors = np.abs(psi_M_denorm_np - gt_np)
            pixel_errs.extend(errors.flatten().tolist())
            img_maes.append(np.mean(errors))
            p, s = calculate_psnr_ssim_global(gt_np, psi_M_denorm_np)
            if not np.isnan(p): img_psnrs.append(p)
            if not np.isnan(s): img_ssims.append(s)
            
    return pixel_errs, img_maes, img_psnrs, img_ssims


# --- dWC Metóda: Dataset, Evaluácia ---
class CustomDataset_dWC_Eval(Dataset):
    def __init__(self, path_to_data, norm_stats_input, k_max_val, target_img_size):
        self.path = path_to_data
        self.image_files_wrapped_all = sorted(glob.glob(os.path.join(self.path, 'images', "wrappedbg_*.tiff")))
        self.image_files_wrapped = []
        self.label_files_unwrapped_gt = []
        for img_path in self.image_files_wrapped_all:
            base_id_name = os.path.basename(img_path).replace('wrappedbg_', '').replace('.tiff', '')
            lbl_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'labels', f'unwrapped_{base_id_name}.tiff')
            if os.path.exists(lbl_path):
                self.image_files_wrapped.append(img_path)
                self.label_files_unwrapped_gt.append(lbl_path)
        self.image_files_wrapped_all = None
        if not self.image_files_wrapped: raise FileNotFoundError(f"Nenašli sa platné páry v {path_to_data} pre dWC eval")
        self.input_min_g, self.input_max_g = norm_stats_input
        self.k_max = k_max_val
        self.target_img_size = target_img_size
        self.current_img_path_for_debug = "N/A"

    def _normalize_input_minmax(self, data, min_val, max_val):
        if max_val == min_val: return torch.zeros_like(data) if isinstance(data, torch.Tensor) else np.zeros_like(data)
        return 2.0 * (data - min_val) / (max_val - min_val) - 1.0

    def __len__(self): return len(self.image_files_wrapped)

    def __getitem__(self, index):
        wrapped_img_path = self.image_files_wrapped[index]
        unwrapped_gt_path = self.label_files_unwrapped_gt[index]
        self.current_img_path_for_debug = wrapped_img_path
        try:
            wrapped_orig_np = tifffile.imread(wrapped_img_path)
            unwrapped_gt_orig_np = tifffile.imread(unwrapped_gt_path)
        except Exception as e:
            print(f"CHYBA (dWC Dataset __getitem__): {e} pri {wrapped_img_path}"); return None, None, None
        
        wrapped_for_input_np = _ensure_shape_global(wrapped_orig_np.copy(), self.target_img_size, f"dWC_Input ({os.path.basename(wrapped_img_path)})")
        unwrapped_gt_for_eval_np = _ensure_shape_global(unwrapped_gt_orig_np.copy(), self.target_img_size, f"dWC_GT ({os.path.basename(unwrapped_gt_path)})")
        
        if wrapped_for_input_np is None or unwrapped_gt_for_eval_np is None : return None, None, None

        wrapped_input_norm_np = self._normalize_input_minmax(wrapped_for_input_np, self.input_min_g, self.input_max_g)
        wrapped_input_norm_tensor = torch.from_numpy(wrapped_input_norm_np.astype(np.float32)).unsqueeze(0)
        
        return wrapped_input_norm_tensor, \
               wrapped_for_input_np.astype(np.float32), \
               unwrapped_gt_for_eval_np.astype(np.float32)

def collate_fn_dWC_eval(batch):
    batch = list(filter(lambda x: x is not None and all(item is not None for item in x), batch))
    if not batch: return None, None, None
    try: return torch.utils.data.dataloader.default_collate(batch)
    except Exception as e: print(f"CHYBA v collate_fn_dWC_eval: {e}"); return None, None, None

def evaluate_dWC_method(config_path_clf, weights_path_clf, test_dataset_path, device_str):
    config = {}; 
    with open(config_path_clf, 'r') as f:
        for line in f:
            if ":" in line: key, value = line.split(":", 1); config[key.strip()] = value.strip()
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
        else:
            print(f"VAROVANIE (dWC): Nepodarilo sa parsovať Min/Max z '{input_norm_str}'. Používam default.")
    except Exception as e_parse:
        print(f"CHYBA parsovania Min/Max pre dWC: {e_parse}. Používam default.")


    dataset = CustomDataset_dWC_Eval(test_dataset_path, (global_input_min, global_input_max), k_max_val, TARGET_IMG_SIZE)
    if len(dataset) == 0: print("dWC: Testovací dataset prázdny."); return [], [], [], []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn_dWC_eval)

    device = torch.device(device_str); print(f"dWC používa: {device}")
    model = smp.Unet(encoder_name=encoder_name, encoder_weights=None, in_channels=1, classes=num_classes_effective, activation=None).to(device)
    try:
        model.load_state_dict(torch.load(weights_path_clf, map_location=device, weights_only=True))
    except Exception as e: print(f"CHYBA nacitania vah dWC: {e}"); return [],[],[],[]
    model.eval()

    pixel_errs, img_maes, img_psnrs, img_ssims = [], [], [], []
    print("Spracovanie dWC modelu (rekonštrukcia fázy)...")
    with torch.no_grad():
        for data_batch in tqdm(loader, desc="dWC Eval"):
            if data_batch is None or data_batch[0] is None: continue
            input_norm_tensor, wrapped_orig_np_batch, gt_unwrapped_np_batch = data_batch
            
            if input_norm_tensor is None : continue

            logits = model(input_norm_tensor.to(device))
            # Výstup siete má shape (Batch=1, NumClasses, H, W)
            # Predikované triedy (H,W)
            pred_classes = torch.argmax(logits.squeeze(0), dim=0).cpu() 
            
            k_pred_values = pred_classes.float() - k_max_val # (H,W)
            
            wrapped_orig_np = wrapped_orig_np_batch.cpu().numpy().squeeze() # (H,W)
            gt_unwrapped_np = gt_unwrapped_np_batch.cpu().numpy().squeeze()   # (H,W)

            reconstructed_phase = wrapped_orig_np + (2 * np.pi) * k_pred_values.numpy() # (H,W)
            
            errors = np.abs(reconstructed_phase - gt_unwrapped_np)
            pixel_errs.extend(errors.flatten().tolist())
            img_maes.append(np.mean(errors))
            p, s = calculate_psnr_ssim_global(gt_unwrapped_np, reconstructed_phase)
            if not np.isnan(p): img_psnrs.append(p)
            if not np.isnan(s): img_ssims.append(s)
            
    return pixel_errs, img_maes, img_psnrs, img_ssims

# --- L2 Metóda (Skimage) ---
def evaluate_L2_method(test_dataset_path):
    images_dir = os.path.join(test_dataset_path, 'images')
    labels_dir = os.path.join(test_dataset_path, 'labels')
    image_files_wrapped = sorted(glob.glob(os.path.join(images_dir, "wrappedbg_*.tiff")))
    pixel_errors_l2, image_maes_l2, image_psnrs_l2, image_ssims_l2 = [], [], [], []
    print("Spracovanie L2 (skimage.restoration.unwrap_phase)...")
    for img_path in tqdm(image_files_wrapped, desc="L2 Eval"):
        base_id_name = os.path.basename(img_path).replace('wrappedbg_', '').replace('.tiff', '')
        lbl_path = os.path.join(labels_dir, f'unwrapped_{base_id_name}.tiff')
        if not os.path.exists(lbl_path): continue
        try:
            wrapped_img_orig = tifffile.imread(img_path)
            gt_unwrapped_img_orig = tifffile.imread(lbl_path)

            wrapped_img = _ensure_shape_global(wrapped_img_orig, TARGET_IMG_SIZE, f"L2_Input ({os.path.basename(img_path)})")
            gt_unwrapped_img = _ensure_shape_global(gt_unwrapped_img_orig, TARGET_IMG_SIZE, f"L2_GT ({os.path.basename(lbl_path)})")
            if wrapped_img is None or gt_unwrapped_img is None: continue

            unwrapped_l2_orig = skimage_unwrap_phase(wrapped_img)
            unwrapped_l2 = _ensure_shape_global(unwrapped_l2_orig, TARGET_IMG_SIZE, f"L2_Output ({os.path.basename(img_path)})")
            if unwrapped_l2 is None : continue
            
            errors = np.abs(unwrapped_l2 - gt_unwrapped_img)
            pixel_errors_l2.extend(errors.flatten().tolist())
            image_maes_l2.append(np.mean(errors))
            p, s = calculate_psnr_ssim_global(gt_unwrapped_img, unwrapped_l2)
            if not np.isnan(p): image_psnrs_l2.append(p)
            if not np.isnan(s): image_ssims_l2.append(s)
        except Exception as e:
            print(f"Chyba pri L2 pre {img_path}: {e}")
    return pixel_errors_l2, image_maes_l2, image_psnrs_l2, image_ssims_l2

# --- Goldsteinova Metóda ---
def evaluate_goldstein_method(goldstein_results_path, gt_labels_path):
    print(f"Spracovanie predvypočítaných výsledkov Goldstein (GS) metódy...")
    pixel_errors_gs, image_maes_gs, image_psnrs_gs, image_ssims_gs = [], [], [], []
    
    if not os.path.isdir(goldstein_results_path):
        print(f"CHYBA: Adresár '{goldstein_results_path}' s výsledkami Goldstein metódy neexistuje.")
        return [], [], [], []
    if not os.path.isdir(gt_labels_path):
        print(f"CHYBA: Adresár '{gt_labels_path}' s GT labelmi neexistuje.")
        return [], [], [], []

    gt_files = sorted(glob.glob(os.path.join(gt_labels_path, "unwrapped_*.tiff")))
    if not gt_files: print(f"CHYBA: Nenašli sa GT súbory v {gt_labels_path}"); return [], [], [], []
    
    processed_count_gs = 0
    for gt_path in tqdm(gt_files, desc="Goldstein Eval"):
        try:
            base_filename = os.path.basename(gt_path)
            gs_pred_path = os.path.join(goldstein_results_path, base_filename)

            if not os.path.exists(gs_pred_path): continue

            gt_unwrapped_img_orig = tifffile.imread(gt_path)
            gs_unwrapped_img_orig = tifffile.imread(gs_pred_path)
            
            gt_unwrapped_img = _ensure_shape_global(gt_unwrapped_img_orig, TARGET_IMG_SIZE, f"GS_GT ({base_filename})")
            gs_unwrapped_img = _ensure_shape_global(gs_unwrapped_img_orig, TARGET_IMG_SIZE, f"GS_Pred ({base_filename})")
            if gt_unwrapped_img is None or gs_unwrapped_img is None: continue
            
            errors = np.abs(gs_unwrapped_img - gt_unwrapped_img)
            pixel_errors_gs.extend(errors.flatten().tolist())
            image_maes_gs.append(np.mean(errors))
            p, s = calculate_psnr_ssim_global(gt_unwrapped_img, gs_unwrapped_img)
            if not np.isnan(p): image_psnrs_gs.append(p)
            if not np.isnan(s): image_ssims_gs.append(s)
            processed_count_gs +=1
        except Exception as e:
            print(f"Chyba pri GS pre {gt_path}: {e}")
    if processed_count_gs == 0 and len(gt_files) > 0 :
         print(f"VAROVANIE (GS): Neboli spracované žiadne súbory.")
    return pixel_errors_gs, image_maes_gs, image_psnrs_gs, image_ssims_gs

# --- Hlavný skript ---
if __name__ == '__main__':
    CONFIG_PATH_DRG = r"C:\Users\viera\Desktop\q_tiff\TRENOVANIE_bakalarka_simul\optimalizacia_hype\trenovanie_5\config_R34imgnet_direct_MAE_GDL0p3_AugMedium_LR5em04_WD1em04_Ep120_ESp30_Tmax120_EtaMin1em07_bs8_bs8.txt" 
    WEIGHTS_PATH_DRG = r"C:\Users\viera\Desktop\q_tiff\TRENOVANIE_bakalarka_simul\optimalizacia_hype\trenovanie_5\best_weights_R34imgnet_direct_MAE_GDL0p3_AugMedium_LR5em04_WD1em04_Ep120_ESp30_Tmax120_EtaMin1em07_bs8_bs8.pth"
    CONFIG_PATH_DWC = r"C:\Users\viera\Desktop\q_tiff\TRENOVANIE_bakalarka_simul\classification\experiment4_hyper\config_clf_R34imgnet_Kmax6_AugMed_LR1em03_WD1em04_Ep120_Tmax120_EtaMin1em07_EdgeW5.0_bs8.txt"
    WEIGHTS_PATH_DWC = r"C:\Users\viera\Desktop\q_tiff\TRENOVANIE_bakalarka_simul\classification\experiment4_hyper\best_weights_clf_R34imgnet_Kmax6_AugMed_LR1em03_WD1em04_Ep120_Tmax120_EtaMin1em07_EdgeW5.0_bs8.pth"
    TEST_DATA_PATH_ROOT = r'C:\Users\viera\Desktop\q_tiff\split_dataset_tiff_for_dynamic_v_stratified_final\static_test_dataset'
    GOLDSTEIN_RESULTS_PATH = r"C:\\Users\\viera\\Desktop\\matlab_goldstain\\2d-phase-unwrap-goldstein\\goldstein_unwrapped"
    OUTPUT_COMPARISON_DIR = r"C:\Users\viera\Desktop\GRAAFY\POROVNANIE_METOD_FINALNE_V3"
    DEVICE_TO_USE = 'cuda'

    start_time_overall = time.time()
    all_methods_results_data = {} 

    print("\n--- Evaluácia dRG modelu ---")
    if os.path.exists(CONFIG_PATH_DRG) and os.path.exists(WEIGHTS_PATH_DRG) and os.path.isdir(TEST_DATA_PATH_ROOT):
        dRG_pix_err, dRG_maes, dRG_psnrs, dRG_ssims = evaluate_dRG_method(CONFIG_PATH_DRG, WEIGHTS_PATH_DRG, TEST_DATA_PATH_ROOT, DEVICE_TO_USE)
        all_methods_results_data['dRG'] = {'pixel_errors': dRG_pix_err, 'image_maes': dRG_maes, 'image_psnrs': dRG_psnrs, 'image_ssims': dRG_ssims}
    else: print(f"CHYBA: Chýbajú súbory/cesta pre dRG. Preskakujem."); all_methods_results_data['dRG'] = {'pixel_errors': [], 'image_maes': [], 'image_psnrs': [], 'image_ssims': []}

    print("\n--- Evaluácia dWC modelu ---")
    if os.path.exists(CONFIG_PATH_DWC) and os.path.exists(WEIGHTS_PATH_DWC) and os.path.isdir(TEST_DATA_PATH_ROOT):
        dWC_pix_err, dWC_maes, dWC_psnrs, dWC_ssims = evaluate_dWC_method(CONFIG_PATH_DWC, WEIGHTS_PATH_DWC, TEST_DATA_PATH_ROOT, DEVICE_TO_USE)
        all_methods_results_data['dWC'] = {'pixel_errors': dWC_pix_err, 'image_maes': dWC_maes, 'image_psnrs': dWC_psnrs, 'image_ssims': dWC_ssims}
    else: print(f"CHYBA: Chýbajú súbory/cesta pre dWC. Preskakujem."); all_methods_results_data['dWC'] = {'pixel_errors': [], 'image_maes': [], 'image_psnrs': [], 'image_ssims': []}

    print("\n--- Evaluácia L2 (skimage) ---")
    if os.path.isdir(TEST_DATA_PATH_ROOT):
        L2_pix_err, L2_maes, L2_psnrs, L2_ssims = evaluate_L2_method(TEST_DATA_PATH_ROOT)
        all_methods_results_data['L2'] = {'pixel_errors': L2_pix_err, 'image_maes': L2_maes, 'image_psnrs': L2_psnrs, 'image_ssims': L2_ssims}
    else: print(f"CHYBA: Chýba cesta pre L2 dáta: {TEST_DATA_PATH_ROOT}"); all_methods_results_data['L2'] = {'pixel_errors': [], 'image_maes': [], 'image_psnrs': [], 'image_ssims': []}
    
    print("\n--- Evaluácia Goldstein (GS) ---")
    gt_labels_path_for_gs = os.path.join(TEST_DATA_PATH_ROOT, "labels") 
    if os.path.isdir(GOLDSTEIN_RESULTS_PATH) and os.path.isdir(gt_labels_path_for_gs): # Použijeme gt_labels_path_for_gs
         GS_pix_err, GS_maes, GS_psnrs, GS_ssims = evaluate_goldstein_method(GOLDSTEIN_RESULTS_PATH, gt_labels_path_for_gs) # Odstránený TARGET_IMG_SIZE, už je v _ensure_shape_global
         all_methods_results_data['GS'] = {'pixel_errors': GS_pix_err, 'image_maes': GS_maes, 'image_psnrs': GS_psnrs, 'image_ssims': GS_ssims}
    else:
        print(f"CHYBA: Adresár s výsledkami Goldstein ('{GOLDSTEIN_RESULTS_PATH}') alebo GT labels ('{gt_labels_path_for_gs}') neexistuje. Preskakujem GS.")
        all_methods_results_data['GS'] = {'pixel_errors': [], 'image_maes': [], 'image_psnrs': [], 'image_ssims': []}

    # --- Generovanie Histogramov ---
    print("\nGenerujem histogramy chýb...")
    os.makedirs(OUTPUT_COMPARISON_DIR, exist_ok=True)
    
    # fig_hist_grid, axs_hist_grid = plt.subplots(2, 2, figsize=(15, 12))
    # fig_hist_grid.suptitle("Porovnanie histogramov absolútnych pixelových chýb", fontsize=18, y=0.97)
    
    # plot_positions = [(0,0), (0,1), (1,0), (1,1)]
    method_names_ordered = ['dRG', 'dWC', 'L2', 'GS'] 
    # hist_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 

    # for i_method, method_key in enumerate(method_names_ordered):
    #     ax = axs_hist_grid[plot_positions[i_method]]
    #     if method_key in all_methods_results_data and all_methods_results_data[method_key]['pixel_errors']:
    #         pixel_errors_data = all_methods_results_data[method_key]['pixel_errors']
    #         if not pixel_errors_data: 
    #             ax.text(0.5, 0.5, f"{method_key.upper()}\n(Žiadne dáta)", ha='center', va='center', transform=ax.transAxes, fontsize=10)
    #         else:
    #             ax.hist(pixel_errors_data, bins=100, alpha=0.8, edgecolor='black', linewidth=0.5, color=hist_colors[i_method % len(hist_colors)])
    #             ax.set_yscale('log')
    #             ax.set_title(f"Histogram absolútnych chýb {method_key.upper()}", fontsize=14)
    #         ax.set_xlabel("Absolútna chyba [rad]", fontsize=11)
    #         ax.set_ylabel("Počet pixelov (log)", fontsize=11)
    #         ax.tick_params(axis='both', which='major', labelsize=9)
    #         ax.grid(True, linestyle=':', alpha=0.6)
    #     else:
    #         ax.text(0.5, 0.5, f"{method_key.upper()}\n(Dáta nedostupné)", ha='center', va='center', transform=ax.transAxes, fontsize=10)
    #         ax.set_xlabel("Absolútna chyba [rad]", fontsize=11)
    #         ax.set_ylabel("Počet pixelov (log)", fontsize=11)
    #         ax.tick_params(axis='both', which='major', labelsize=9)

    # plt.tight_layout(rect=[0, 0.03, 1, 0.94]) 
    # plt.savefig(os.path.join(OUTPUT_COMPARISON_DIR, "comparison_histograms_grid.png"), dpi=300)
    # plt.savefig(os.path.join(OUTPUT_COMPARISON_DIR, "comparison_histograms_grid.svg"))
    # print(f"Spoločný graf histogramov uložený do: {OUTPUT_COMPARISON_DIR}")
    # plt.close(fig_hist_grid)

    # --- Štatistiky do súboru ---
    print("\nGenerujem štatistiky...")
    stats_output_path = os.path.join(OUTPUT_COMPARISON_DIR, "all_methods_metrics_summary.txt")
    with open(stats_output_path, 'w', encoding='utf-8') as f:
        f.write("Súhrnné štatistiky pre porovnávané metódy rozbalenia fázy\n")
        f.write("="*60 + "\n\n")
        for method_key in method_names_ordered:
            f.write(f"--- Metóda: {method_key.upper()} ---\n")
            data = all_methods_results_data.get(method_key, {})
            maes = np.array([m for m in data.get('image_maes', []) if not np.isnan(m)])
            psnrs = np.array([p for p in data.get('image_psnrs', []) if not np.isnan(p) and not np.isinf(p)]) 
            ssims = np.array([s for s in data.get('image_ssims', []) if not np.isnan(s)])

            if len(maes) > 0:
                f.write(f"MAE (na obrázok):\n")
                f.write(f"  Priemer: {np.mean(maes):.6f} rad\n")
                f.write(f"  Medián: {np.median(maes):.6f} rad\n")
                f.write(f"  Std.odch: {np.std(maes):.6f} rad\n")
                f.write(f"  Min: {np.min(maes):.8f} rad\n")
                f.write(f"  Max: {np.max(maes):.8f} rad\n")
            else: f.write(f"MAE: Žiadne platné dáta.\n")
            
            if len(psnrs) > 0:
                f.write(f"PSNR (na obrázok):\n")
                f.write(f"  Priemer: {np.mean(psnrs):.6f} dB\n")
                f.write(f"  Medián: {np.median(psnrs):.6f} dB\n")
                f.write(f"  Std.odch: {np.std(psnrs):.6f} dB\n")
                f.write(f"  Min: {np.min(psnrs):.8f} dB\n")
                f.write(f"  Max: {np.max(psnrs):.8f} dB\n")
            else: f.write(f"PSNR: Žiadne platné dáta.\n")

            if len(ssims) > 0:
                f.write(f"SSIM (na obrázok):\n")
                f.write(f"  Priemer: {np.mean(ssims):.6f}\n")
                f.write(f"  Medián: {np.median(ssims):.6f}\n")
                f.write(f"  Std.odch: {np.std(ssims):.6f}\n")
                f.write(f"  Min: {np.min(ssims):.8f}\n")
                f.write(f"  Max: {np.max(ssims):.8f}\n")
            else: f.write(f"SSIM: Žiadne platné dáta.\n")
            f.write("\n")
    print(f"Súhrnné štatistiky uložené do: {stats_output_path}")

    # --- Boxploty MAE ---
    print("\nGenerujem boxploty MAE...")
    plot_data_boxplot = []
    plot_labels_boxplot = []
    for method_key in method_names_ordered:
        if method_key in all_methods_results_data and all_methods_results_data[method_key]['image_maes']:
            valid_maes_for_boxplot = [m for m in all_methods_results_data[method_key]['image_maes'] if not np.isnan(m)]
            if valid_maes_for_boxplot:
                plot_data_boxplot.append(valid_maes_for_boxplot)
                plot_labels_boxplot.append(method_key.upper())
    
    if plot_data_boxplot:
        plt.figure(figsize=(10, 7))
        bp = plt.boxplot(plot_data_boxplot,
                         labels=plot_labels_boxplot,
                         patch_artist=True,
                         medianprops={'linewidth': 2, 'color': 'firebrick'},
                         showfliers=True,
                         notch=False)
        
        boxplot_colors = ['#1f77b4', "#0ef3ff", '#2ca02c', "#d3d627"] 
        for patch_idx, patch in enumerate(bp['boxes']):
            patch.set_facecolor(boxplot_colors[patch_idx % len(boxplot_colors)])
            patch.set_alpha(0.7) 
            patch.set_edgecolor('black')

        for whisker in bp['whiskers']:
            whisker.set(color='black', linewidth=1.2, linestyle='--')
        for cap in bp['caps']:
            cap.set(color='black', linewidth=1.2)
        for flier in bp['fliers']:
            flier.set(marker='o', color='black', alpha=0.5, markersize=6, markerfacecolor='lightgray')
        # Medianprops je už nastavený pri volaní plt.boxplot, ale ak by ste chceli dodatočne:
        # for median in bp['medians']: 
        #     median.set(linewidth=2.5) # Príklad, ak by bolo treba


        plt.title("Porovnanie distribúcie MAE medzi metódami", fontsize=16, pad=15)
        plt.ylabel("MAE [rad]", fontsize=12)
        plt.xticks(fontsize=11, rotation=0) 
        plt.yticks(fontsize=11)
        plt.grid(True, linestyle=':', alpha=0.6, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_COMPARISON_DIR, "comparison_mae_boxplots.png"), dpi=300)
        plt.savefig(os.path.join(OUTPUT_COMPARISON_DIR, "comparison_mae_boxplots.svg"))
        print(f"Boxploty MAE uložené do: {OUTPUT_COMPARISON_DIR}")
        # plt.show() 
        plt.close()
    else:
        print("Nemožno vytvoriť boxploty MAE, chýbajú dáta.")

    # --- Boxploty PSNR ---
    print("\nGenerujem boxploty PSNR...")
    plot_data_boxplot_psnr = []
    plot_labels_boxplot_psnr = []
    for method_key in method_names_ordered:
        if method_key in all_methods_results_data and all_methods_results_data[method_key]['image_psnrs']:
            valid_psnrs_for_boxplot = [p for p in all_methods_results_data[method_key]['image_psnrs'] if not np.isnan(p) and not np.isinf(p)]
            if valid_psnrs_for_boxplot:
                plot_data_boxplot_psnr.append(valid_psnrs_for_boxplot)
                plot_labels_boxplot_psnr.append(method_key.upper())
    
    if plot_data_boxplot_psnr:
        plt.figure(figsize=(10, 7))
        bp_psnr = plt.boxplot(plot_data_boxplot_psnr,
                              labels=plot_labels_boxplot_psnr,
                              patch_artist=True,
                              medianprops={'linewidth': 2, 'color': 'darkgreen'}, # Iná farba pre medián
                              showfliers=True,
                              notch=False)
        
        # Použijeme rovnaké farby boxov ako pre MAE pre konzistenciu, alebo môžete definovať nové
        boxplot_colors_psnr = ['#1f77b4', "#0ef3ff", '#2ca02c', "#d3d627"]
        for patch_idx, patch in enumerate(bp_psnr['boxes']):
            patch.set_facecolor(boxplot_colors_psnr[patch_idx % len(boxplot_colors_psnr)])
            patch.set_alpha(0.7) 
            patch.set_edgecolor('black')

        for whisker in bp_psnr['whiskers']:
            whisker.set(color='black', linewidth=1.2, linestyle='--')
        for cap in bp_psnr['caps']:
            cap.set(color='black', linewidth=1.2)
        for flier in bp_psnr['fliers']:
            flier.set(marker='o', color='black', alpha=0.5, markersize=6, markerfacecolor='lightgray')

        plt.title("Porovnanie distribúcie PSNR medzi metódami", fontsize=16, pad=15)
        plt.ylabel("PSNR [dB]", fontsize=12)
        plt.xticks(fontsize=11, rotation=0) 
        plt.yticks(fontsize=11)
        plt.grid(True, linestyle=':', alpha=0.6, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_COMPARISON_DIR, "comparison_psnr_boxplots.png"), dpi=300)
        plt.savefig(os.path.join(OUTPUT_COMPARISON_DIR, "comparison_psnr_boxplots.svg"))
        print(f"Boxploty PSNR uložené do: {OUTPUT_COMPARISON_DIR}")
        plt.close()
    else:
        print("Nemožno vytvoriť boxploty PSNR, chýbajú dáta.")

    # --- Boxploty SSIM ---
    print("\nGenerujem boxploty SSIM...")
    plot_data_boxplot_ssim = []
    plot_labels_boxplot_ssim = []
    for method_key in method_names_ordered:
        if method_key in all_methods_results_data and all_methods_results_data[method_key]['image_ssims']:
            valid_ssims_for_boxplot = [s for s in all_methods_results_data[method_key]['image_ssims'] if not np.isnan(s)]
            if valid_ssims_for_boxplot:
                plot_data_boxplot_ssim.append(valid_ssims_for_boxplot)
                plot_labels_boxplot_ssim.append(method_key.upper())
                
    if plot_data_boxplot_ssim:
        plt.figure(figsize=(10, 7))
        bp_ssim = plt.boxplot(plot_data_boxplot_ssim,
                              labels=plot_labels_boxplot_ssim,
                              patch_artist=True,
                              medianprops={'linewidth': 2, 'color': 'purple'}, # Iná farba pre medián
                              showfliers=True,
                              notch=False)
        
        boxplot_colors_ssim = ['#1f77b4', "#0ef3ff", '#2ca02c', "#d3d627"]
        for patch_idx, patch in enumerate(bp_ssim['boxes']):
            patch.set_facecolor(boxplot_colors_ssim[patch_idx % len(boxplot_colors_ssim)])
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')

        for whisker in bp_ssim['whiskers']:
            whisker.set(color='black', linewidth=1.2, linestyle='--')
        for cap in bp_ssim['caps']:
            cap.set(color='black', linewidth=1.2)
        for flier in bp_ssim['fliers']:
            flier.set(marker='o', color='black', alpha=0.5, markersize=6, markerfacecolor='lightgray')

        plt.title("Porovnanie distribúcie SSIM medzi metódami", fontsize=16, pad=15)
        plt.ylabel("SSIM na obrázok", fontsize=12)
        plt.xticks(fontsize=11, rotation=0)
        plt.yticks(fontsize=11)
        plt.grid(True, linestyle=':', alpha=0.6, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_COMPARISON_DIR, "comparison_ssim_boxplots.png"), dpi=300)
        plt.savefig(os.path.join(OUTPUT_COMPARISON_DIR, "comparison_ssim_boxplots.svg"))
        print(f"Boxploty SSIM uložené do: {OUTPUT_COMPARISON_DIR}")
        plt.close()
    else:
        print("Nemožno vytvoriť boxploty SSIM, chýbajú dáta.")

    total_script_time_overall = time.time() - start_time_overall
    print(f"\nCelkové spracovanie všetkých skriptov dokončené za {time.strftime('%H:%M:%S', time.gmtime(total_script_time_overall))}.")  
import os
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tifffile as tiff
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as T
import segmentation_models_pytorch as smp
from skimage import io
from skimage.filters import gaussian

# ---------------------------- GLOBALNE KONSTANTY ----------------------------
NORMALIZED_INPUT_CLAMP_MIN = -1.0
NORMALIZED_INPUT_CLAMP_MAX = 1.0
NORMALIZED_INPUT_ERASING_VALUE = 0.0

# Konfiguracia pre dynamicku simulaciu
SIMULATION_PARAM_RANGES = {
    "n_strips_param": (7, 8),
    "original_image_influence_param": (0.3, 0.5),
    "phase_noise_std_param": (0.024, 0.039),
    "smooth_original_image_sigma_param": (0.2, 0.5),
    "poly_scale_param": (0.02, 0.1),
    "CURVATURE_AMPLITUDE_param": (1.4, 2.0),
    "background_offset_d_param": (-24.8, -6.8),
    "tilt_angle_deg_param": (-5.0, 17.0)
}
AMPLIFY_AB_VALUE = 1.0

OUTPUT_BASE_DIR = "split_dataset_tiff_for_dynamic_v_stratified_final"
PATH_STATIC_REF_TRAIN = os.path.join(OUTPUT_BASE_DIR, "static_ref_train_dataset")
PATH_STATIC_VALID = os.path.join(OUTPUT_BASE_DIR, "static_valid_dataset")
PATH_STATIC_TEST = os.path.join(OUTPUT_BASE_DIR, "static_test_dataset")
PATH_DYNAMIC_TRAIN_SOURCE_IMAGES = os.path.join(OUTPUT_BASE_DIR, "train_dataset_source_for_dynamic_generation", "images")

ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = "imagenet"
AUGMENTATION_STRENGTH_TRAIN = "medium"
LOSS_TYPE = "mae_gdl"
LAMBDA_GDL = 0.3
LEARNING_RATE = 5e-4
BATCH_SIZE = 8
NUM_EPOCHS = 120
EARLY_STOPPING_PATIENCE = 30
COSINE_T_MAX = 120
COSINE_ETA_MIN = 1e-7
WEIGHT_DECAY = 1e-4
TARGET_IMG_SIZE = (512, 512)
MODEL_SAVE_PATH = "drg_best_model_final.pth"
CURVES_SAVE_PATH = "drg_training_curves_final.png"
VIS_SAVE_PATH = "drg_test_visualization_final.png"

# ---------------------------- POMOCNE FUNKCIE PRE SIMULACIU ----------------------------
def get_random_or_fixed(param_value, is_integer=False, allow_float_for_int=False):
    if isinstance(param_value, (list, tuple)) and len(param_value) == 2:
        min_val, max_val = param_value
        if min_val > max_val:
            min_val, max_val = max_val, min_val
        if is_integer:
            low = int(round(min_val))
            high = int(round(max_val))
            if high < low: high = low
            if not allow_float_for_int and isinstance(min_val, int) and isinstance(max_val, int):
                return np.random.randint(min_val, max_val + 1)
            else:
                return int(np.random.randint(low, high + 1))
        else: # Float
            return np.random.uniform(min_val, max_val)
    return param_value

def wrap_phase(img: np.ndarray) -> np.ndarray:
    """
    img: Vstupný obrázok fázy.
    """
    return (img + np.pi) % (2 * np.pi) - np.pi

def generate_cubic_background(shape: tuple, n_strips: int, tilt_angle_deg: float, a_val: float, b_val: float, c_val: float, d_val: float) -> np.ndarray:
    """
    shape: Rozmery obrázka (H, W).
    n_strips: Počet prúžkov.
    tilt_angle_deg: Uhol náklonu v stupňoch.
    a_val, b_val, c_val, d_val: Koeficienty kubickej funkcie.
    """
    H, W = shape
    y_idxs, x_idxs = np.indices((H, W))
    slope_y = (n_strips * 2 * np.pi) / H
    tilt_rad = np.deg2rad(tilt_angle_deg)
    slope_x = slope_y * np.tan(tilt_rad)
    linear_grad = slope_y * y_idxs + slope_x * x_idxs
    x_norm = np.linspace(0, 1, W)
    poly1d = a_val * x_norm**3 + b_val * x_norm**2 + c_val * x_norm + d_val
    poly2d = np.tile(poly1d, (H, 1))
    background = linear_grad + poly2d
    return background

def generate_simulation_pair(source_img_np: np.ndarray, param_ranges_config: dict, amplify_ab_value: float) -> tuple[np.ndarray, np.ndarray]:
    """
    source_img_np: Zdrojový obrázok ako numpy pole.
    param_ranges_config: Konfigurácia intervalov pre parametre simulácie.
    amplify_ab_value: Hodnota pre zosilnenie koeficientov a, b.
    """
    n_strips_dyn = get_random_or_fixed(param_ranges_config["n_strips_param"], is_integer=True, allow_float_for_int=True)
    original_image_influence_dyn = get_random_or_fixed(param_ranges_config["original_image_influence_param"])
    phase_noise_std_dyn = get_random_or_fixed(param_ranges_config["phase_noise_std_param"])
    smooth_sigma_dyn = get_random_or_fixed(param_ranges_config["smooth_original_image_sigma_param"])
    
    poly_scale_dyn = get_random_or_fixed(param_ranges_config["poly_scale_param"])
    curvature_amplitude_base_dyn = get_random_or_fixed(param_ranges_config["CURVATURE_AMPLITUDE_param"])
    background_offset_base_dyn = get_random_or_fixed(param_ranges_config["background_offset_d_param"])
    tilt_angle_deg_dyn = get_random_or_fixed(param_ranges_config["tilt_angle_deg_param"])

    # Výpočet koeficientov a,b,c,d na základe dynamických parametrov
    mean_a = 0.0 * curvature_amplitude_base_dyn
    mean_b = -4.0 * curvature_amplitude_base_dyn
    mean_c = +4.0 * curvature_amplitude_base_dyn
    mean_d = background_offset_base_dyn

    # Základné štandardné odchýlky (pred škálovaním pomocou poly_scale_dyn)
    std_base_a = 0.3 * curvature_amplitude_base_dyn
    std_base_b = 0.3 * curvature_amplitude_base_dyn
    std_base_c = 0.3 * curvature_amplitude_base_dyn
    std_base_d = 2.0

    # Vzorkovanie finálnych koeficientov
    a_final = np.random.normal(mean_a, std_base_a * poly_scale_dyn) * amplify_ab_value
    b_final = np.random.normal(mean_b, std_base_b * poly_scale_dyn) * amplify_ab_value
    c_final = np.random.normal(mean_c, std_base_c * poly_scale_dyn)
    d_final = np.random.normal(mean_d, std_base_d * poly_scale_dyn)

    if smooth_sigma_dyn > 0:
        base_img = gaussian(source_img_np, sigma=smooth_sigma_dyn, preserve_range=True)
    else:
        base_img = source_img_np
    object_phase = base_img * (2 * np.pi)
    
    background = generate_cubic_background(source_img_np.shape,
                                           n_strips_dyn,
                                           tilt_angle_deg_dyn,
                                           a_final, b_final, c_final, d_final)
                                           
    unwrapped = object_phase * original_image_influence_dyn + background * (1.0 - original_image_influence_dyn)
    if phase_noise_std_dyn > 0:
        unwrapped += np.random.normal(0, phase_noise_std_dyn, size=source_img_np.shape)
    wrapped = wrap_phase(unwrapped)
    return unwrapped.astype(np.float32), wrapped.astype(np.float32)

# ---------------------------- FUNKCIE PRE GLOBALNE STATISTIKY ----------------------------
def compute_global_stats(dataset_path_for_stats: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    dataset_path_for_stats: Cesta k adresáru datasetu pre výpočet štatistík.
    """
    image_files = sorted(glob.glob(os.path.join(dataset_path_for_stats, 'images', "*.tiff")))
    label_files = sorted(glob.glob(os.path.join(dataset_path_for_stats, 'labels', "*.tiff")))
    
    if not image_files: print(f"Varovanie: Nenajdene obrazky v {os.path.join(dataset_path_for_stats, 'images')}")
    if not label_files: print(f"Varovanie: Nenajdene labely v {os.path.join(dataset_path_for_stats, 'labels')}")

    input_mins, input_maxs = [], []
    for f_img in image_files:
        img = tiff.imread(f_img).astype(np.float32)
        input_mins.append(img.min())
        input_maxs.append(img.max())
    
    global_input_min = min(input_mins) if input_mins else 0.0
    global_input_max = max(input_maxs) if input_maxs else 1.0

    target_vals_flat = []
    for f_lbl in label_files:
        lbl = tiff.imread(f_lbl).astype(np.float32)
        target_vals_flat.append(lbl.flatten())
    
    if target_vals_flat:
        all_target_vals = np.concatenate(target_vals_flat)
        global_target_mean = np.mean(all_target_vals)
        global_target_std = np.std(all_target_vals)
    else:
        global_target_mean = 0.0
        global_target_std = 1.0
        
    if global_input_min == global_input_max: global_input_max += 1e-6
    if global_target_std < 1e-6: global_target_std = 1.0

    print(f"Globalne statistiky (z {dataset_path_for_stats}):")
    print(f"  Wrapped Input: Min={global_input_min:.4f}, Max={global_input_max:.4f}")
    print(f"  Unwrapped Target: Mean={global_target_mean:.4f}, Std={global_target_std:.4f}")
    return (global_input_min, global_input_max), (global_target_mean, global_target_std)

# ---------------------------- AUGMENTACNA TRANSFORMACIA ----------------------------
class AddGaussianNoiseTransform(nn.Module):
    def __init__(self, std_dev_range: tuple[float, float] = (0.03, 0.12), p: float = 0.5, clamp_min: float = None, clamp_max: float = None):
        """
        std_dev_range: Rozsah (min, max) pre štandardnú odchýlku šumu.
        p: Pravdepodobnosť aplikácie transformácie.
        clamp_min: Minimálna hodnota pre orezanie po pridaní šumu.
        clamp_max: Maximálna hodnota pre orezanie po pridaní šumu.
        """
        super().__init__()
        self.std_dev_min, self.std_dev_max = std_dev_range
        self.p = p
        self.clamp_min, self.clamp_max = clamp_min, clamp_max
    def forward(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        img_tensor: Vstupný tenzor obrázka.
        """
        if torch.rand(1).item() < self.p:
            std_dev = torch.empty(1).uniform_(self.std_dev_min, self.std_dev_max).item()
            noise = torch.randn_like(img_tensor) * std_dev
            noisy_img = img_tensor + noise
            if self.clamp_min is not None and self.clamp_max is not None:
                noisy_img = torch.clamp(noisy_img, self.clamp_min, self.clamp_max)
            return noisy_img
        return img_tensor

# ---------------------------- DATASETY ----------------------------
class StaticPhaseDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_data: str, norm_stats_input_global: tuple[float, float], norm_stats_target_global: tuple[float, float], target_img_size: tuple[int, int] = (512,512)):
        """
        path_to_data: Cesta k adresáru so statickými dátami.
        norm_stats_input_global: Globálne (min, max) pre normalizáciu vstupu.
        norm_stats_target_global: Globálne (mean, std) pre normalizáciu cieľa.
        target_img_size: Cieľová veľkosť obrázkov (H, W).
        """
        self.path = path_to_data
        self.image_list = sorted(glob.glob(os.path.join(self.path, 'images', "*.tiff")))
        self.input_min_g, self.input_max_g = norm_stats_input_global
        self.target_mean_g, self.target_std_g = norm_stats_target_global
        self.target_img_size = target_img_size

    def _normalize_input(self, data: torch.Tensor) -> torch.Tensor:
        return 2.0 * (data - self.input_min_g) / (self.input_max_g - self.input_min_g) - 1.0

    def _normalize_target(self, data: torch.Tensor) -> torch.Tensor:
        return (data - self.target_mean_g) / self.target_std_g
    
    def __len__(self): return len(self.image_list)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_list[index]
        base_id_name = os.path.basename(img_path).replace('wrappedbg_', '').replace('.tiff','')
        lbl_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'labels', f'unwrapped_{base_id_name}.tiff')

        wrapped_orig_phase = tiff.imread(img_path).astype(np.float32)
        unwrapped_orig = tiff.imread(lbl_path).astype(np.float32)
        
        wrapped_tensor = torch.from_numpy(wrapped_orig_phase.copy()).unsqueeze(0)
        unwrapped_tensor = torch.from_numpy(unwrapped_orig.copy()).unsqueeze(0)

        return self._normalize_input(wrapped_tensor), self._normalize_target(unwrapped_tensor)

class DynamicTrainPhaseDataset(torch.utils.data.Dataset):
    def __init__(self, source_image_filepaths: list[str], 
                 simulation_param_ranges: dict, 
                 amplify_ab_value: float,
                 norm_stats_input_global: tuple[float, float], 
                 norm_stats_target_global: tuple[float, float], 
                 augmentation_strength: str = 'none', 
                 target_img_size: tuple[int, int] = (512,512)):
        """
        source_image_filepaths: Zoznam ciest k zdrojovým obrázkom.
        simulation_param_ranges: Konfigurácia intervalov pre parametre simulácie.
        amplify_ab_value: Hodnota pre zosilnenie koeficientov a, b.
        norm_stats_input_global: Globálne (min, max) pre normalizáciu vstupu.
        norm_stats_target_global: Globálne (mean, std) pre normalizáciu cieľa.
        augmentation_strength: Sila augmentácie ('none', 'light', 'medium', 'strong').
        target_img_size: Cieľová veľkosť obrázkov (H, W).
        """
        self.source_image_filepaths = source_image_filepaths
        self.simulation_param_ranges = simulation_param_ranges
        self.amplify_ab_value = amplify_ab_value
        self.input_min_g, self.input_max_g = norm_stats_input_global
        self.target_mean_g, self.target_std_g = norm_stats_target_global
        self.target_img_size = target_img_size
        self.augmentation_strength = augmentation_strength
        self.geometric_transforms = None
        self.pixel_transforms = None
        if self.augmentation_strength != 'none':
            self._setup_augmentations(self.augmentation_strength)

    def _setup_augmentations(self, strength: str):
        noise_std_range, noise_p = (0.01, 0.05), 0.0
        erase_scale, erase_p = (0.01, 0.04), 0.0
        if strength == 'light': noise_std_range, noise_p, erase_scale, erase_p = (0.02, 0.08), 0.4, (0.01,0.05), 0.3
        elif strength == 'medium': noise_std_range, noise_p, erase_scale, erase_p = (0.03, 0.12), 0.5, (0.02,0.08), 0.4
        elif strength == 'strong': noise_std_range, noise_p, erase_scale, erase_p = (0.05, 0.15), 0.6, (0.02,0.10), 0.5
        
        self.geometric_transforms = T.Compose([T.RandomHorizontalFlip(p=0.5)])
        pixel_aug_list = []
        if noise_p > 0: pixel_aug_list.append(AddGaussianNoiseTransform(noise_std_range, noise_p, NORMALIZED_INPUT_CLAMP_MIN, NORMALIZED_INPUT_CLAMP_MAX))
        if erase_p > 0: pixel_aug_list.append(T.RandomErasing(p=erase_p, scale=erase_scale, ratio=(0.3, 3.3), value=NORMALIZED_INPUT_ERASING_VALUE))
        if pixel_aug_list: self.pixel_transforms = T.Compose(pixel_aug_list)

    def _normalize_input(self, data: torch.Tensor) -> torch.Tensor:
        return 2.0 * (data - self.input_min_g) / (self.input_max_g - self.input_min_g) - 1.0

    def _normalize_target(self, data: torch.Tensor) -> torch.Tensor:
        return (data - self.target_mean_g) / self.target_std_g

    def __len__(self): return len(self.source_image_filepaths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        source_img_raw = io.imread(self.source_image_filepaths[idx]).astype(np.float32)
        s_min, s_max = source_img_raw.min(), source_img_raw.max()
        source_img_norm = (source_img_raw - s_min) / (s_max - s_min) if s_max > s_min else np.zeros_like(source_img_raw)
        
        # Pouzitie dynamickej simulacie s konfiguracnymi parametrami
        unwrapped_phase_np, wrapped_phase_np = generate_simulation_pair(source_img_norm, self.simulation_param_ranges, self.amplify_ab_value)
        
        wrapped_tensor = torch.from_numpy(wrapped_phase_np.copy()).unsqueeze(0)
        unwrapped_tensor = torch.from_numpy(unwrapped_phase_np.copy()).unsqueeze(0)

        norm_wrapped = self._normalize_input(wrapped_tensor)
        norm_unwrapped = self._normalize_target(unwrapped_tensor)

        if self.geometric_transforms:
            stacked = torch.cat((norm_wrapped, norm_unwrapped), dim=0)
            stacked_transformed = self.geometric_transforms(stacked)
            norm_wrapped, norm_unwrapped = torch.chunk(stacked_transformed, 2, dim=0)

        if self.pixel_transforms:
            norm_wrapped = self.pixel_transforms(norm_wrapped)
            
        return norm_wrapped, norm_unwrapped

# ---------------------------- LOSS FUNKCIE A DENORMALIZACIA ----------------------------
def mae_loss_norm(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor: 
    """
    p: Predikovaný tenzor.
    t: Cieľový tenzor.
    """
    return torch.mean(torch.abs(p-t))

def mse_loss_norm(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor: 
    """
    p: Predikovaný tenzor.
    t: Cieľový tenzor.
    """
    return F.mse_loss(p,t)

def sobel_gradient_loss(yt: torch.Tensor, yp: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    yt: Cieľový tenzor.
    yp: Predikovaný tenzor.
    device: Zariadenie (cpu/cuda).
    """
    if yt.ndim==3: yt=yt.unsqueeze(1)
    if yp.ndim==3: yp=yp.unsqueeze(1)
    sx_w = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=torch.float32,device=device).view(1,1,3,3)
    sy_w = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=torch.float32,device=device).view(1,1,3,3)
    gx_t=F.conv2d(yt,sx_w,padding=1); gy_t=F.conv2d(yt,sy_w,padding=1)
    gx_p=F.conv2d(yp,sx_w,padding=1); gy_p=F.conv2d(yp,sy_w,padding=1)
    return torch.mean(torch.abs(gx_t-gx_p)) + torch.mean(torch.abs(gy_t-gy_p))

def denormalize_target(data_norm: torch.Tensor, mean_orig: float, std_orig: float) -> torch.Tensor: 
    """
    data_norm: Normalizovaný tenzor.
    mean_orig: Pôvodný priemer.
    std_orig: Pôvodná štandardná odchýlka.
    """
    return data_norm * std_orig + mean_orig

def denormalize_input(data_norm: torch.Tensor, min_orig: float, max_orig: float) -> torch.Tensor:
    """
    data_norm: Normalizovaný tenzor (v rozsahu [-1, 1]).
    min_orig: Pôvodné minimum.
    max_orig: Pôvodné maximum.
    """
    return (data_norm + 1.0) * (max_orig - min_orig) / 2.0 + min_orig

# ---------------------------- HLAVNA TRENOVACIA FUNKCIA ----------------------------
def train_evaluate_model(global_input_stats: tuple[float, float], 
                         global_target_stats: tuple[float, float],
                         sim_param_ranges: dict,
                         sim_amplify_ab: float):
    """
    global_input_stats: Globálne (min, max) pre vstup.
    global_target_stats: Globálne (mean, std) pre cieľ.
    sim_param_ranges: Konfigurácia intervalov pre parametre dynamickej simulácie.
    sim_amplify_ab: Hodnota pre zosilnenie koeficientov a, b v simulácii.
    """
    (input_min_g, input_max_g) = global_input_stats
    (target_mean_g, target_std_g) = global_target_stats
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_source_files = glob.glob(os.path.join(PATH_DYNAMIC_TRAIN_SOURCE_IMAGES, "*.tif*"))
    if not train_source_files: print(f"Varovanie: Nenajdene zdrojove obrazky v {PATH_DYNAMIC_TRAIN_SOURCE_IMAGES}")
    
    train_ds = DynamicTrainPhaseDataset(train_source_files, 
                                        sim_param_ranges,
                                        sim_amplify_ab,
                                        global_input_stats, 
                                        global_target_stats, 
                                        AUGMENTATION_STRENGTH_TRAIN, 
                                        TARGET_IMG_SIZE)
    valid_ds = StaticPhaseDataset(PATH_STATIC_VALID, global_input_stats, global_target_stats, TARGET_IMG_SIZE)
    test_ds = StaticPhaseDataset(PATH_STATIC_TEST, global_input_stats, global_target_stats, TARGET_IMG_SIZE)

    # num_workers je tu explicitne 1, ale pre všeobecnosť ponechávam podmienku
    num_workers_val = 1 
    pin_memory_flag = True if torch.cuda.is_available() else False

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers_val, 
                              persistent_workers=True if num_workers_val > 0 else False,
                              pin_memory=pin_memory_flag)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers_val,
                              persistent_workers=True if num_workers_val > 0 else False,
                              pin_memory=pin_memory_flag)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers_val,
                             persistent_workers=True if num_workers_val > 0 else False,
                             pin_memory=pin_memory_flag)

    model = smp.Unet(encoder_name=ENCODER_NAME, encoder_weights=ENCODER_WEIGHTS, in_channels=1, classes=1, activation=None).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=COSINE_T_MAX, eta_min=COSINE_ETA_MIN)

    train_loss_hist, val_loss_hist, train_mae_denorm_hist, val_mae_denorm_hist = [],[],[],[]
    best_val_mae_denorm = float('inf')
    epochs_no_improve = 0

    print("Zaciatok treningu...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_train_loss, epoch_train_mae_d = [], []
        for inputs_n, targets_n in train_loader:
            inputs_n, targets_n = inputs_n.to(device), targets_n.to(device)
            optimizer.zero_grad()
            preds_n = model(inputs_n)
            
            main_loss = mae_loss_norm(preds_n, targets_n) if 'mae' in LOSS_TYPE else mse_loss_norm(preds_n, targets_n)
            total_loss = main_loss + (LAMBDA_GDL * sobel_gradient_loss(targets_n, preds_n, device) if 'gdl' in LOSS_TYPE else 0)
            total_loss.backward()
            optimizer.step()

            epoch_train_loss.append(total_loss.item())
            with torch.no_grad():
                preds_d = denormalize_target(preds_n, target_mean_g, target_std_g)
                targets_d = denormalize_target(targets_n, target_mean_g, target_std_g)
                epoch_train_mae_d.append(torch.mean(torch.abs(preds_d - targets_d)).item())
        
        train_loss_hist.append(np.mean(epoch_train_loss) if epoch_train_loss else float('nan'))
        train_mae_denorm_hist.append(np.mean(epoch_train_mae_d) if epoch_train_mae_d else float('nan'))

        model.eval()
        epoch_val_loss, epoch_val_mae_d = [], []
        with torch.no_grad():
            for inputs_n_val, targets_n_val in valid_loader:
                inputs_n_val, targets_n_val = inputs_n_val.to(device), targets_n_val.to(device)
                preds_n_val = model(inputs_n_val)
                main_loss_v = mae_loss_norm(preds_n_val, targets_n_val) if 'mae' in LOSS_TYPE else mse_loss_norm(preds_n_val, targets_n_val)
                total_loss_v = main_loss_v + (LAMBDA_GDL * sobel_gradient_loss(targets_n_val, preds_n_val, device) if 'gdl' in LOSS_TYPE else 0)
                epoch_val_loss.append(total_loss_v.item())
                preds_d_val = denormalize_target(preds_n_val, target_mean_g, target_std_g)
                targets_d_val = denormalize_target(targets_n_val, target_mean_g, target_std_g)
                epoch_val_mae_d.append(torch.mean(torch.abs(preds_d_val - targets_d_val)).item())

        avg_val_loss = np.mean(epoch_val_loss) if epoch_val_loss else float('nan')
        avg_val_mae_d = np.mean(epoch_val_mae_d) if epoch_val_mae_d else float('nan')
        val_loss_hist.append(avg_val_loss)
        val_mae_denorm_hist.append(avg_val_mae_d)

        print(f"Ep {epoch+1}/{NUM_EPOCHS} | Tr L: {train_loss_hist[-1]:.4f}, Tr MAE(D): {train_mae_denorm_hist[-1]:.4f} | Val L: {avg_val_loss:.4f}, Val MAE(D): {avg_val_mae_d:.4f} | LR: {optimizer.param_groups[0]['lr']:.1e}")

        if not np.isnan(avg_val_mae_d) and avg_val_mae_d < best_val_mae_denorm:
            best_val_mae_denorm = avg_val_mae_d
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  Novy najlepsi Val MAE(D): {best_val_mae_denorm:.4f}. Model ulozeny.")
            epochs_no_improve = 0
        elif not np.isnan(avg_val_mae_d):
            epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping @ epocha {epoch+1}.")
            break
        scheduler.step()
    
    print(f"Trening dokonceny. Najlepsie Val MAE(D): {best_val_mae_denorm:.4f}")

    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        model.eval()
        test_mae_d_list = []
        print("\nTestovanie modelu...")
        with torch.no_grad():
            for i, (inputs_n_test, targets_n_test) in enumerate(test_loader):
                inputs_n_test, targets_n_test = inputs_n_test.to(device), targets_n_test.to(device)
                preds_n_test = model(inputs_n_test)
                preds_d_test = denormalize_target(preds_n_test, target_mean_g, target_std_g)
                targets_d_test = denormalize_target(targets_n_test, target_mean_g, target_std_g)
                test_mae_d_list.append(torch.mean(torch.abs(preds_d_test - targets_d_test)).item())
                if i == 0: 
                    in_vis = denormalize_input(inputs_n_test[0].cpu(), input_min_g, input_max_g).numpy().squeeze()
                    pred_vis = preds_d_test[0].cpu().numpy().squeeze()
                    gt_vis = targets_d_test[0].cpu().numpy().squeeze()
                    fig_vis, axes_vis = plt.subplots(1,3,figsize=(15,5))
                    axes_vis[0].imshow(in_vis, cmap='gray'); axes_vis[0].set_title("Vstup (Denorm)")
                    axes_vis[1].imshow(gt_vis, cmap='gray'); axes_vis[1].set_title("Ground Truth (Denorm)")
                    axes_vis[2].imshow(pred_vis, cmap='gray'); axes_vis[2].set_title("Predikcia (Denorm)")
                    plt.tight_layout(); plt.savefig(VIS_SAVE_PATH); plt.close(fig_vis)
        avg_test_mae_d = np.mean(test_mae_d_list) if test_mae_d_list else float('nan')
        print(f"Priemerne Test MAE (Denorm): {avg_test_mae_d:.4f}")

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1); plt.plot(train_loss_hist, label='Train Loss (N)'); plt.plot(val_loss_hist, label='Val Loss (N)'); plt.legend(); plt.title("Normalizovana Strata")
    plt.subplot(1,2,2); plt.plot(train_mae_denorm_hist, label='Train MAE (D)'); plt.plot(val_mae_denorm_hist, label='Val MAE (D)'); plt.legend(); plt.title("Denormalizovana MAE")
    plt.tight_layout(); plt.savefig(CURVES_SAVE_PATH); plt.show()

# ---------------------------- SPUSTENIE ----------------------------
if __name__ == '__main__':
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    paths_to_check = [PATH_STATIC_REF_TRAIN, PATH_STATIC_VALID, PATH_STATIC_TEST, PATH_DYNAMIC_TRAIN_SOURCE_IMAGES]
    for p in paths_to_check:
        if not os.path.exists(p):
            print(f"CHYBA: Adresar '{p}' neexistuje. Skontrolujte cesty k datasetom.")
            exit()
            
    input_stats, target_stats = compute_global_stats(PATH_STATIC_REF_TRAIN)
    # Preposlanie konfiguracii pre dynamicku simulaciu do hlavnej funkcie
    train_evaluate_model(input_stats, target_stats, SIMULATION_PARAM_RANGES, AMPLIFY_AB_VALUE)
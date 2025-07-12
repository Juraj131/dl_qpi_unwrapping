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
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as T_v2 # Pouzivame v2 pre augmentacie ak su potrebne
import segmentation_models_pytorch as smp
from skimage import io
from skimage.filters import gaussian

# ---------------------------- GLOBALNE KONSTANTY ----------------------------
NORMALIZED_INPUT_CLAMP_MIN = -1.0
NORMALIZED_INPUT_CLAMP_MAX = 1.0
NORMALIZED_INPUT_ERASING_VALUE = 0.0
KMAX_DEFAULT_CALCULATED = 6

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

# Staticke hyperparametre pre DWC model
ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = "imagenet"
AUGMENTATION_STRENGTH_TRAIN = "medium"
LEARNING_RATE = 1e-3
BATCH_SIZE = 8
NUM_EPOCHS = 120
EARLY_STOPPING_PATIENCE = 30
COSINE_T_MAX = 120
COSINE_ETA_MIN = 1e-7
WEIGHT_DECAY = 1e-4
EDGE_LOSS_WEIGHT = 5.0
TARGET_IMG_SIZE = (512, 512)
MODEL_SAVE_PATH = "dwc_best_model_final.pth"
CURVES_SAVE_PATH = "dwc_training_curves_final.png"
VIS_K_SAVE_PATH = "dwc_test_k_labels_final.png"
VIS_RECON_SAVE_PATH = "dwc_test_reconstruction_final.png"

# ---------------------------- POMOCNE FUNKCIE PRE SIMULACIU (z dwc_optimalization) ----------------------------
def get_random_or_fixed(param_value, is_integer=False, allow_float_for_int=False):
    if isinstance(param_value, (list, tuple)) and len(param_value) == 2:
        min_val, max_val = param_value
        if min_val > max_val: min_val, max_val = max_val, min_val
        if is_integer:
            low, high = int(round(min_val)), int(round(max_val))
            if high < low: high = low
            return int(np.random.randint(low, high + 1))
        else:
            return np.random.uniform(min_val, max_val)
    return param_value

def wrap_phase(img: np.ndarray) -> np.ndarray:
    """
    img: Vstupny obrazok fazy.
    """
    return (img + np.pi) % (2 * np.pi) - np.pi

def generate_cubic_background(shape: tuple, coeff_stats: list, scale: float =1.0, amplify_ab: float =1.0, n_strips: int =6, tilt_angle_deg: float =0.0) -> tuple[np.ndarray, dict]:
    """
    shape: Rozmery obrazka (H, W).
    coeff_stats: Statistiky pre koeficienty [ (mean_a, std_a), ... ].
    scale: Skala pre standardne odchylky koeficientov.
    amplify_ab: Zosilnenie pre koeficienty a, b.
    n_strips: Pocet pruzkov.
    tilt_angle_deg: Uhol naklonu v stupnoch.
    """
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
    a_val *= amplify_ab; b_val *= amplify_ab
    poly1d = a_val * x_norm**3 + b_val * x_norm**2 + c_val * x_norm + d_val
    poly2d = np.tile(poly1d, (H, 1))
    background = linear_grad + poly2d
    return background

def generate_simulation_pair_from_source_np(source_image_np: np.ndarray, param_ranges_config: dict, amplify_ab_value: float) -> tuple[np.ndarray, np.ndarray]:
    """
    source_image_np: Normalizovany zdrojovy obrazok (0-1).
    param_ranges_config: Konfiguracia intervalov pre parametre simulacie.
    amplify_ab_value: Hodnota pre zosilnenie koeficientov a, b.
    """
    n_strips = get_random_or_fixed(param_ranges_config["n_strips_param"], is_integer=True)
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
    if smooth_original_image_sigma > 0 and original_image_influence > 0:
        img_phase_obj_base = gaussian(source_image_np, sigma=smooth_original_image_sigma, preserve_range=True, channel_axis=None)
    else:
        img_phase_obj_base = source_image_np
    object_phase_contribution = img_phase_obj_base * (2 * np.pi)
    
    generated_background = generate_cubic_background(
        source_image_np.shape, coeff_stats_for_bg_generation,
        scale=poly_scale, amplify_ab=amplify_ab_value,
        n_strips=n_strips, tilt_angle_deg=tilt_angle_deg
    )
    unwrapped_phase = (object_phase_contribution * original_image_influence) + \
                      (generated_background * (1.0 - original_image_influence))
    if phase_noise_std > 0:
        unwrapped_phase += np.random.normal(0, phase_noise_std, size=source_image_np.shape)
    wrapped_phase = wrap_phase(unwrapped_phase)
    return unwrapped_phase.astype(np.float32), wrapped_phase.astype(np.float32)

# ---------------------------- FUNKCIE PRE GLOBALNE STATISTIKY (DWC SPECIFICKE) ----------------------------
def calculate_wrapped_input_min_max(dataset_path: str) -> tuple[float, float]:
    """
    dataset_path: Cesta k adresaru datasetu pre vypocet Min/Max pre wrapped vstup.
    """
    image_files = glob.glob(os.path.join(dataset_path, 'images', "*.tiff"))
    if not image_files:
        print(f"Varovanie: Nenasli sa ziadne obrazky v {os.path.join(dataset_path, 'images')} pre Min/Max.")
        return -np.pi, np.pi # Fallback

    min_val_g, max_val_g = np.inf, -np.inf
    print(f"Pocitam Min/Max pre wrapped vstup z {len(image_files)} suborov v {dataset_path}...")
    for i, fp in enumerate(image_files):
        img = tiff.imread(fp).astype(np.float32)
        min_val_g = min(min_val_g, img.min())
        max_val_g = max(max_val_g, img.max())
    if np.isinf(min_val_g) or np.isinf(max_val_g):
        print("Varovanie: Nepodarilo sa urcit platne Min/Max, pouzivam fallback.")
        return -np.pi, np.pi
    print(f"Vypocet Min/Max dokonceny: Min={min_val_g:.4f}, Max={max_val_g:.4f}")
    return float(min_val_g), float(max_val_g)

def calculate_k_max(dataset_path: str) -> int:
    """
    dataset_path: Cesta k adresaru datasetu pre vypocet k_max.
    """
    images_dir = os.path.join(dataset_path, 'images')
    labels_dir = os.path.join(dataset_path, 'labels')
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.tiff")))
    if not image_files:
        print(f"Varovanie: Nenasli sa ziadne obrazky v {images_dir} pre k_max.")
        return KMAX_DEFAULT_CALCULATED 

    max_abs_k_found = 0.0
    print(f"Pocitam k_max z {len(image_files)} parov v {dataset_path}...")
    for img_path in image_files:
        base_id_name = os.path.basename(img_path).replace('wrappedbg_', '').replace('.tiff', '')
        lbl_path = os.path.join(labels_dir, f'unwrapped_{base_id_name}.tiff')
        if not os.path.exists(lbl_path): continue

        wrapped_orig_np = tiff.imread(img_path).astype(np.float32)
        unwrapped_orig_np = tiff.imread(lbl_path).astype(np.float32)
        if wrapped_orig_np.shape != unwrapped_orig_np.shape: continue

        k_diff_np = (unwrapped_orig_np - wrapped_orig_np) / (2 * np.pi)
        k_float_np = np.round(k_diff_np)
        current_max_abs_k = np.max(np.abs(k_float_np))
        if current_max_abs_k > max_abs_k_found:
            max_abs_k_found = current_max_abs_k
            
    calculated_k_max = int(np.ceil(max_abs_k_found))
    if calculated_k_max == 0 and max_abs_k_found == 0: # Ak su vsetky k=0
        print(f"Varovanie: Vypocitane k_max je 0. Pouzivam default: {KMAX_DEFAULT_CALCULATED}")
        return KMAX_DEFAULT_CALCULATED
    print(f"Vypocet k_max dokonceny. Najdene max_abs_k_float: {max_abs_k_found:.2f}, Vypocitane k_max_val: {calculated_k_max}")
    return calculated_k_max

# ---------------------------- AUGMENTACNA TRANSFORMACIA (z dwc_optimalization) ----------------------------
class AddGaussianNoiseTransform(nn.Module):
    def __init__(self, std_dev_range=(0.03, 0.12), p=0.5, clamp_min=None, clamp_max=None):
        """
        std_dev_range: Rozsah (min, max) pre standardnu odchylku sumu.
        p: Pravdepodobnost aplikacie transformacie.
        clamp_min: Minimalna hodnota pre orezanie po pridani sumu.
        clamp_max: Maximalna hodnota pre orezanie po pridani sumu.
        """
        super().__init__()
        self.std_dev_min, self.std_dev_max = std_dev_range
        self.p = p
        self.clamp_min, self.clamp_max = clamp_min, clamp_max
    def forward(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        img_tensor: Vstupny tenzor obrazka.
        """
        if torch.rand(1).item() < self.p:
            std_dev = torch.empty(1).uniform_(self.std_dev_min, self.std_dev_max).item()
            noise = torch.randn_like(img_tensor) * std_dev
            noisy_img = img_tensor + noise
            if self.clamp_min is not None and self.clamp_max is not None:
                noisy_img = torch.clamp(noisy_img, self.clamp_min, self.clamp_max)
            return noisy_img
        return img_tensor

# ---------------------------- DATASETY (DWC SPECIFICKE) ----------------------------
class WrapCountDatasetBase(Dataset):
    def __init__(self, input_min_max_global, k_max_val, target_img_size, edge_loss_weight):
        self.input_min_g, self.input_max_g = input_min_max_global
        self.k_max = k_max_val
        self.target_img_size = target_img_size
        self.edge_loss_weight = edge_loss_weight

    def _normalize_input_minmax_to_minus_one_one(self, data: torch.Tensor) -> torch.Tensor:
        if self.input_max_g == self.input_min_g: return torch.zeros_like(data)
        return 2.0 * (data - self.input_min_g) / (self.input_max_g - self.input_min_g) - 1.0

    def _ensure_shape_and_type(self, img_numpy: np.ndarray, data_name: str ="Image") -> np.ndarray:
        img_numpy = img_numpy.astype(np.float32)
        if img_numpy.shape[-2:] != self.target_img_size:
            pass
        return img_numpy
    
    def _process_pair(self, wrapped_np: np.ndarray, unwrapped_np: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        wrapped_tensor = torch.from_numpy(wrapped_np.copy()).unsqueeze(0)
        unwrapped_tensor = torch.from_numpy(unwrapped_np.copy()).unsqueeze(0)

        diff = (unwrapped_tensor - wrapped_tensor) / (2 * np.pi)
        k_float = torch.round(diff)
        k_float = torch.clamp(k_float, -self.k_max, self.k_max)
        k_label = (k_float + self.k_max).long().squeeze(0) # (H, W)

        wrapped_input_norm = self._normalize_input_minmax_to_minus_one_one(wrapped_tensor.squeeze(0)).unsqueeze(0) # (1, H, W)

        k_label_np = k_label.cpu().numpy()
        edge_mask = np.zeros_like(k_label_np, dtype=bool)
        edge_mask[:, :-1] |= (k_label_np[:, :-1] != k_label_np[:, 1:])
        edge_mask[:, 1:] |= (k_label_np[:, 1:] != k_label_np[:, :-1])
        edge_mask[:-1, :] |= (k_label_np[:-1, :] != k_label_np[1:, :])
        edge_mask[1:, :] |= (k_label_np[1:, :] != k_label_np[:-1, :])
        weight_map_np = np.ones_like(k_label_np, dtype=np.float32)
        weight_map_np[edge_mask] = self.edge_loss_weight
        weight_map_tensor = torch.from_numpy(weight_map_np) # (H,W)

        return wrapped_input_norm, k_label, unwrapped_tensor.squeeze(0), wrapped_tensor.squeeze(0), weight_map_tensor


class StaticWrapCountDataset(WrapCountDatasetBase):
    def __init__(self, path_to_data: str, input_min_max_global: tuple[float, float], k_max_val: int, target_img_size: tuple[int,int], edge_loss_weight: float):
        """
        path_to_data: Cesta k adresaru so statickymi datami.
        input_min_max_global: Globalne (min, max) pre normalizaciu wrapped vstupu.
        k_max_val: Maximalna absolutna hodnota k.
        target_img_size: Cielova velkost obrazkov (H, W).
        edge_loss_weight: Vaha pre pixely na hranach oblasti k.
        """
        super().__init__(input_min_max_global, k_max_val, target_img_size, edge_loss_weight)
        self.path = path_to_data
        self.image_list = sorted(glob.glob(os.path.join(self.path, 'images', "*.tiff")))

    def __len__(self): return len(self.image_list)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path = self.image_list[index]
        base_id_name = os.path.basename(img_path).replace('wrappedbg_', '').replace('.tiff','')
        lbl_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'labels', f'unwrapped_{base_id_name}.tiff')

        wrapped_orig_np = tiff.imread(img_path)
        unwrapped_orig_np = tiff.imread(lbl_path)

        wrapped_orig_np = self._ensure_shape_and_type(wrapped_orig_np, "Wrapped (static)")
        unwrapped_orig_np = self._ensure_shape_and_type(unwrapped_orig_np, "Unwrapped (static)")
        
        return self._process_pair(wrapped_orig_np, unwrapped_orig_np)

class DynamicTrainWrapCountDataset(WrapCountDatasetBase):
    def __init__(self, source_image_filepaths: list[str], sim_param_ranges: dict, amplify_ab_val: float,
                 input_min_max_global: tuple[float, float], k_max_val: int, aug_strength: str, target_img_size: tuple[int,int], edge_loss_weight: float):
        """
        source_image_filepaths: Zoznam ciest k zdrojovym obrazkom.
        sim_param_ranges: Konfiguracia intervalov pre parametre simulacie.
        amplify_ab_val: Hodnota pre zosilnenie koeficientov a, b.
        input_min_max_global: Globalne (min, max) pre normalizaciu wrapped vstupu.
        k_max_val: Maximalna absolutna hodnota k.
        aug_strength: Sila augmentacie ('none', 'light', 'medium', 'strong').
        target_img_size: Cielova velkost obrazkov (H, W).
        edge_loss_weight: Vaha pre pixely na hranach oblasti k.
        """
        super().__init__(input_min_max_global, k_max_val, target_img_size, edge_loss_weight)
        self.source_image_filepaths = source_image_filepaths
        self.simulation_param_ranges = sim_param_ranges
        self.amplify_ab_value = amplify_ab_val
        self.augmentation_strength = aug_strength
        self.geometric_transforms = None
        self.pixel_transforms_for_input = None
        if self.augmentation_strength != 'none':
            self._setup_augmentations(self.augmentation_strength)

    def _setup_augmentations(self, strength: str):
        noise_std_range, noise_p = (0.01, 0.05), 0.0
        erase_scale, erase_p = (0.01, 0.04), 0.0
        if strength == 'light': noise_std_range, noise_p, erase_scale, erase_p = (0.02, 0.08), 0.4, (0.01,0.05), 0.3
        elif strength == 'medium': noise_std_range, noise_p, erase_scale, erase_p = (0.03, 0.12), 0.5, (0.02,0.08), 0.4
        elif strength == 'strong': noise_std_range, noise_p, erase_scale, erase_p = (0.05, 0.15), 0.6, (0.02,0.10), 0.5
        
        geo_transforms_list = []
        if strength in ['light', 'medium', 'strong']: geo_transforms_list.append(T_v2.RandomHorizontalFlip(p=0.5))
        if geo_transforms_list: self.geometric_transforms = T_v2.Compose(geo_transforms_list)

        pixel_aug_list = []
        if noise_p > 0: pixel_aug_list.append(AddGaussianNoiseTransform(noise_std_range, noise_p, NORMALIZED_INPUT_CLAMP_MIN, NORMALIZED_INPUT_CLAMP_MAX))
        if erase_p > 0: pixel_aug_list.append(T_v2.RandomErasing(p=erase_p, scale=erase_scale, ratio=(0.3, 3.3), value=NORMALIZED_INPUT_ERASING_VALUE, inplace=False))
        if pixel_aug_list: self.pixel_transforms_for_input = T_v2.Compose(pixel_aug_list)

    def __len__(self): return len(self.source_image_filepaths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        source_img_path = self.source_image_filepaths[idx]
        source_img_raw = io.imread(source_img_path).astype(np.float32)
        source_img_raw = self._ensure_shape_and_type(source_img_raw, "Source Image (Dynamic)")

        img_min_val, img_max_val = source_img_raw.min(), source_img_raw.max()
        source_img_norm_for_sim = (source_img_raw - img_min_val) / (img_max_val - img_min_val) if img_max_val > img_min_val else np.zeros_like(source_img_raw)

        unwrapped_phase_np, wrapped_phase_np = generate_simulation_pair_from_source_np(
            source_img_norm_for_sim, self.simulation_param_ranges, self.amplify_ab_value
        )
        
        # Prevod na tenzory pred augmentaciami
        wrapped_tensor_orig = torch.from_numpy(wrapped_phase_np.copy()).unsqueeze(0)
        unwrapped_tensor_orig = torch.from_numpy(unwrapped_phase_np.copy()).unsqueeze(0)

        # Geometricke augmentacie (ak su definovane) na oboch fazach sucasne
        if self.geometric_transforms:
            stacked_phases = torch.cat((wrapped_tensor_orig, unwrapped_tensor_orig), dim=0)
            stacked_phases_aug = self.geometric_transforms(stacked_phases)
            wrapped_tensor_aug, unwrapped_tensor_aug = stacked_phases_aug[0:1], stacked_phases_aug[1:2]
        else:
            wrapped_tensor_aug, unwrapped_tensor_aug = wrapped_tensor_orig, unwrapped_tensor_orig

        # Vypocet k_label a weight_map z (potencialne augmentovanych) wrapped/unwrapped
        wrapped_input_norm, k_label, unwrapped_final, wrapped_final, weight_map = self._process_pair(
            wrapped_tensor_aug.squeeze(0).numpy(), unwrapped_tensor_aug.squeeze(0).numpy()
        )
        
        # Pixelove augmentacie (ak su definovane) len na normalizovanom vstupe
        if self.pixel_transforms_for_input:
            wrapped_input_norm = self.pixel_transforms_for_input(wrapped_input_norm)
            
        return wrapped_input_norm, k_label, unwrapped_final, wrapped_final, weight_map

# ---------------------------- COLLATE FUNKCIA (DWC SPECIFICKA) ----------------------------
def collate_fn_skip_none_dwc(batch):
    """
    batch: Zoznam vzoriek z datasetu.
    """
    batch = list(filter(lambda x: all(item is not None for item in x[:5]), batch)) # Kontroluje prvych 5 poloziek
    if not batch: return None, None, None, None, None
    return torch.utils.data.dataloader.default_collate(batch)

# ---------------------------- LOSS A METRIKY (DWC SPECIFICKE) ----------------------------
def cross_entropy_loss_weighted(logits: torch.Tensor, klabels: torch.Tensor, weight_map: torch.Tensor) -> torch.Tensor:
    """
    logits: Vystup modelu (B, NumClasses, H, W).
    klabels: Ground truth k-triedy (B, H, W).
    weight_map: Vahy pre pixely (B, H, W).
    """
    pixel_losses = F.cross_entropy(logits, klabels, reduction='none') 
    weighted_losses = pixel_losses * weight_map
    return torch.mean(weighted_losses)

def k_label_accuracy(logits: torch.Tensor, klabels: torch.Tensor) -> torch.Tensor:
    """
    logits: Vystup modelu.
    klabels: Ground truth k-triedy.
    """
    pred_classes = torch.argmax(logits, dim=1)
    correct = (pred_classes == klabels).float().sum()
    total = klabels.numel()
    return correct / total if total > 0 else torch.tensor(0.0)

# ---------------------------- HLAVNA TRENOVACIA FUNKCIA (DWC) ----------------------------
def train_evaluate_dwc_model(global_input_min_max: tuple[float, float], k_max_calculated: int):
    """
    global_input_min_max: Globalne (min, max) pre wrapped vstup.
    k_max_calculated: Vypocitana maximalna absolutna hodnota k.
    """
    input_min_g, input_max_g = global_input_min_max
    k_max_val = k_max_calculated
    num_classes_effective = 2 * k_max_val + 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Pouzivam k_max_val = {k_max_val}, pocet tried = {num_classes_effective}")
    print(f"Pouzivam globalne Min/Max pre vstup: Min={input_min_g:.4f}, Max={input_max_g:.4f}")

    # DataLoaders
    train_source_files = glob.glob(os.path.join(PATH_DYNAMIC_TRAIN_SOURCE_IMAGES, "*.tif*"))
    if not train_source_files: print(f"Varovanie: Nenajdene zdrojove obrazky v {PATH_DYNAMIC_TRAIN_SOURCE_IMAGES}")
    
    train_ds = DynamicTrainWrapCountDataset(
        train_source_files, SIMULATION_PARAM_RANGES, AMPLIFY_AB_VALUE,
        global_input_min_max, k_max_val, AUGMENTATION_STRENGTH_TRAIN, TARGET_IMG_SIZE, EDGE_LOSS_WEIGHT
    )
    valid_ds = StaticWrapCountDataset(
        PATH_STATIC_VALID, global_input_min_max, k_max_val, TARGET_IMG_SIZE, EDGE_LOSS_WEIGHT
    )
    test_ds = StaticWrapCountDataset(
        PATH_STATIC_TEST, global_input_min_max, k_max_val, TARGET_IMG_SIZE, EDGE_LOSS_WEIGHT
    )

    # Nastavenia pre DataLoader
    num_workers_val = 1 # Ako je pouzite v povodnom kode
    pin_memory_flag = True if torch.cuda.is_available() else False
    persistent_workers_flag = True if num_workers_val > 0 else False

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=num_workers_val, collate_fn=collate_fn_skip_none_dwc,
                              pin_memory=pin_memory_flag, persistent_workers=persistent_workers_flag)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, 
                              num_workers=num_workers_val, collate_fn=collate_fn_skip_none_dwc,
                              pin_memory=pin_memory_flag, persistent_workers=persistent_workers_flag)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, 
                             num_workers=num_workers_val, collate_fn=collate_fn_skip_none_dwc,
                             pin_memory=pin_memory_flag, persistent_workers=persistent_workers_flag)

    # Model, Optimizer, Scheduler
    model = smp.Unet(encoder_name=ENCODER_NAME, encoder_weights=ENCODER_WEIGHTS, 
                     in_channels=1, classes=num_classes_effective, activation=None).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=COSINE_T_MAX, eta_min=COSINE_ETA_MIN)

    train_ce_loss_hist, val_ce_loss_hist = [], []
    train_k_acc_hist, val_k_acc_hist = [], []
    val_mae_recon_hist = []
    best_val_k_accuracy = 0.0
    epochs_no_improve = 0

    print("Zaciatok DWC treningu...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_train_ce, epoch_train_k_acc = [], []
        for batch_idx, batch_data in enumerate(train_loader):
            if batch_data[0] is None: continue
            wrapped_norm, k_labels_gt, _, _, weight_maps = batch_data
            wrapped_norm, k_labels_gt, weight_maps = wrapped_norm.to(device), k_labels_gt.to(device), weight_maps.to(device)

            optimizer.zero_grad()
            logits = model(wrapped_norm)
            loss = cross_entropy_loss_weighted(logits, k_labels_gt, weight_maps)
            loss.backward(); optimizer.step()
            
            with torch.no_grad(): acc = k_label_accuracy(logits, k_labels_gt)
            epoch_train_ce.append(loss.item())
            epoch_train_k_acc.append(acc.item())
        
        train_ce_loss_hist.append(np.mean(epoch_train_ce) if epoch_train_ce else float('nan'))
        train_k_acc_hist.append(np.mean(epoch_train_k_acc) if epoch_train_k_acc else float('nan'))

        model.eval()
        epoch_val_ce, epoch_val_k_acc, epoch_val_mae_recon = [], [], []
        with torch.no_grad():
            for batch_data_val in valid_loader:
                if batch_data_val[0] is None: continue
                wrapped_norm_v, k_labels_gt_v, unwrapped_gt_orig_v, wrapped_orig_v, weight_maps_v = batch_data_val
                wrapped_norm_v, k_labels_gt_v = wrapped_norm_v.to(device), k_labels_gt_v.to(device)
                unwrapped_gt_orig_v, wrapped_orig_v = unwrapped_gt_orig_v.to(device), wrapped_orig_v.to(device)
                weight_maps_v = weight_maps_v.to(device)

                logits_v = model(wrapped_norm_v)
                epoch_val_ce.append(cross_entropy_loss_weighted(logits_v, k_labels_gt_v, weight_maps_v).item())
                epoch_val_k_acc.append(k_label_accuracy(logits_v, k_labels_gt_v).item())

                pred_classes_v = torch.argmax(logits_v, dim=1) 
                k_pred_values_v = pred_classes_v.float() - k_max_val 
                unwrapped_pred_reconstructed_v = wrapped_orig_v + (2 * np.pi) * k_pred_values_v
                mae_denorm_v = torch.mean(torch.abs(unwrapped_pred_reconstructed_v - unwrapped_gt_orig_v))
                epoch_val_mae_recon.append(mae_denorm_v.item())
        
        avg_val_ce = np.mean(epoch_val_ce) if epoch_val_ce else float('nan')
        avg_val_k_acc = np.mean(epoch_val_k_acc) if epoch_val_k_acc else float('nan')
        avg_val_mae_recon = np.mean(epoch_val_mae_recon) if epoch_val_mae_recon else float('nan')
        val_ce_loss_hist.append(avg_val_ce)
        val_k_acc_hist.append(avg_val_k_acc)
        val_mae_recon_hist.append(avg_val_mae_recon)

        print(f"Ep {epoch+1}/{NUM_EPOCHS} | Tr CE: {train_ce_loss_hist[-1]:.4f}, Tr kAcc: {train_k_acc_hist[-1]:.4f} | Val CE: {avg_val_ce:.4f}, Val kAcc: {avg_val_k_acc:.4f}, Val MAE(R): {avg_val_mae_recon:.4f} | LR: {optimizer.param_groups[0]['lr']:.1e}")

        if not np.isnan(avg_val_k_acc) and avg_val_k_acc > best_val_k_accuracy:
            best_val_k_accuracy = avg_val_k_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  Novy najlepsi Val k-Acc: {best_val_k_accuracy:.4f}. Model ulozeny.")
            epochs_no_improve = 0
        elif not np.isnan(avg_val_k_acc):
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping @ epocha {epoch+1}.")
            break
        scheduler.step()
    
    print(f"DWC Trening dokonceny. Najlepsie Val k-Accuracy: {best_val_k_accuracy:.4f}")

    # Testovanie
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        model.eval()
        test_k_acc_list, test_mae_recon_list = [], []
        print("\nTestovanie DWC modelu...")
        with torch.no_grad():
            for i, batch_data_test in enumerate(test_loader):
                if batch_data_test[0] is None: continue
                wrapped_norm_t, k_labels_gt_t, unwrapped_gt_orig_t, wrapped_orig_t, _ = batch_data_test
                wrapped_norm_t, k_labels_gt_t = wrapped_norm_t.to(device), k_labels_gt_t.to(device)
                unwrapped_gt_orig_t, wrapped_orig_t = unwrapped_gt_orig_t.to(device), wrapped_orig_t.to(device)

                logits_t = model(wrapped_norm_t)
                test_k_acc_list.append(k_label_accuracy(logits_t, k_labels_gt_t).item())
                
                pred_classes_t = torch.argmax(logits_t, dim=1)
                k_pred_values_t = pred_classes_t.float() - k_max_val
                unwrapped_pred_reconstructed_t = wrapped_orig_t + (2 * np.pi) * k_pred_values_t
                mae_denorm_t = torch.mean(torch.abs(unwrapped_pred_reconstructed_t - unwrapped_gt_orig_t))
                test_mae_recon_list.append(mae_denorm_t.item())

        avg_test_k_acc = np.mean(test_k_acc_list) if test_k_acc_list else float('nan')
        avg_test_mae_recon = np.mean(test_mae_recon_list) if test_mae_recon_list else float('nan')
        print(f"Test k-Accuracy: {avg_test_k_acc:.4f}, Test MAE(R): {avg_test_mae_recon:.4f}")

        # Ulozenie kriviek
        plt.figure(figsize=(18, 5))
        plt.subplot(1, 3, 1)
        plt.plot(train_ce_loss_hist, label='Tréning CE Loss')
        plt.plot(val_ce_loss_hist, label='Validačná CE Loss')
        plt.title('Priebeh Cross-Entropy Loss')
        plt.xlabel('Epocha'); plt.ylabel('CE Loss'); plt.legend(); plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(train_k_acc_hist, label='Tréning k-Accuracy')
        plt.plot(val_k_acc_hist, label='Validačná k-Accuracy')
        plt.title('Priebeh k-Label Accuracy')
        plt.xlabel('Epocha'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(val_mae_recon_hist, label='Validačné MAE (Rekonštrukcia)')
        plt.title('Priebeh Validačného MAE')
        plt.xlabel('Epocha'); plt.ylabel('MAE'); plt.legend(); plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(CURVES_SAVE_PATH)
        plt.close()
        print(f"Grafy priebehu uložene do {CURVES_SAVE_PATH}")

        # Vizualizacia na testovacich datach (prvy batch)
        if test_loader and len(test_loader) > 0:
            model.eval()
            with torch.no_grad():
                first_batch_test = next(iter(test_loader))
                if first_batch_test[0] is not None:
                    wrapped_norm_t, k_labels_gt_t, unwrapped_gt_orig_t, wrapped_orig_t, _ = first_batch_test
                    wrapped_norm_t = wrapped_norm_t.to(device)
                    
                    logits_t_vis = model(wrapped_norm_t)
                    pred_classes_t_vis = torch.argmax(logits_t_vis, dim=1)

                    # Vizualizacia k-tried
                    fig_k, axes_k = plt.subplots(1, 3, figsize=(18, 6))
                    fig_k.suptitle(f"DWC Test - Vizualizácia k-tried (prvý batch)", fontsize=16)
                    
                    idx_to_show = 0 # Zobrazime prvy obrazok z batchu
                    num_classes_effective_vis = 2 * k_max_val + 1

                    axes_k[0].imshow(wrapped_orig_t[idx_to_show].cpu().numpy().squeeze(), cmap='gray')
                    axes_k[0].set_title(f"Vstup Wrapped (orig.)", fontsize=14)
                    
                    axes_k[1].imshow(k_labels_gt_t[idx_to_show].cpu().numpy().squeeze(), cmap='gray', vmin=0, vmax=num_classes_effective_vis-1)
                    axes_k[1].set_title(f"GT k-triedy (0 až {num_classes_effective_vis-1})", fontsize=14)
                    
                    axes_k[2].imshow(pred_classes_t_vis[idx_to_show].cpu().numpy().squeeze(), cmap='gray', vmin=0, vmax=num_classes_effective_vis-1)
                    axes_k[2].set_title(f"Predikované k-triedy (0 až {num_classes_effective_vis-1})", fontsize=14)
                    
                    plt.tight_layout(rect=[0,0,1,0.95])
                    plt.savefig(VIS_K_SAVE_PATH)
                    plt.close(fig_k)
                    print(f"Vizualizácia k-tried uložená do {VIS_K_SAVE_PATH}")

                    # Vizualizacia rekonstrukcie
                    k_pred_values_t_vis = pred_classes_t_vis.float() - k_max_val
                    unwrapped_pred_reconstructed_t_vis = wrapped_orig_t.cpu() + (2 * np.pi) * k_pred_values_t_vis.cpu()

                    fig_r, axes_r = plt.subplots(1, 3, figsize=(18, 6))
                    fig_r.suptitle(f"DWC Test - Vizualizácia Rekonštrukcie (prvý batch)", fontsize=16)

                    im0 = axes_r[0].imshow(wrapped_orig_t[idx_to_show].cpu().numpy().squeeze(), cmap='gray')
                    axes_r[0].set_title("Vstup Wrapped (orig.)")
                    fig_r.colorbar(im0, ax=axes_r[0], fraction=0.046, pad=0.04)

                    im1 = axes_r[1].imshow(unwrapped_gt_orig_t[idx_to_show].cpu().numpy().squeeze(), cmap='gray')
                    axes_r[1].set_title("GT Unwrapped (orig.)")
                    fig_r.colorbar(im1, ax=axes_r[1], fraction=0.046, pad=0.04)

                    im2 = axes_r[2].imshow(unwrapped_pred_reconstructed_t_vis[idx_to_show].cpu().numpy().squeeze(), cmap='gray')
                    axes_r[2].set_title("Predikovaná Rekonštrukcia")
                    fig_r.colorbar(im2, ax=axes_r[2], fraction=0.046, pad=0.04)

                    plt.tight_layout(rect=[0,0,1,0.95])
                    plt.savefig(VIS_RECON_SAVE_PATH)
                    plt.close(fig_r)
                    print(f"Vizualizácia rekonštrukcie uložená do {VIS_RECON_SAVE_PATH}")
    else:
        print(f"Model {MODEL_SAVE_PATH} nebol nájdený. Testovanie preskočené.")


if __name__ == "__main__":
    print("--- Štart DWC Best Model Script ---")
    
    # Vypocet globalnych statistik
    print("\n--- Výpočet Globálnych Štatistík ---")
    # 1. Min/Max pre wrapped vstup z referencneho treningoveho datasetu
    global_wrapped_min, global_wrapped_max = calculate_wrapped_input_min_max(PATH_STATIC_REF_TRAIN)
    current_global_input_min_max = (global_wrapped_min, global_wrapped_max)
    print(f"Globálne Min/Max pre wrapped vstup: ({global_wrapped_min:.4f}, {global_wrapped_max:.4f})")

    # 2. k_max z referencneho treningoveho datasetu
    current_k_max_val = calculate_k_max(PATH_STATIC_REF_TRAIN)
    print(f"Vypočítané k_max_val: {current_k_max_val}")
    
    # Spustenie hlavnej funkcie treningu a evaluacie
    print("\n--- Štart Tréningu a Evaluácie DWC Modelu ---")
    train_evaluate_dwc_model(
        global_input_min_max=current_global_input_min_max,
        k_max_calculated=current_k_max_val
    )
    
    print("\n--- DWC Best Model Script Dokončený ---")

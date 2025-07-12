import os
import glob
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tifffile as tiff
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as T
import segmentation_models_pytorch as smp
from skimage import io
from skimage.filters import gaussian

# 1. GLOBALNE KONSTANTY A IMPORTY
KMAX_DEFAULT = 6
NORMALIZED_INPUT_CLAMP_MIN = -1.0
NORMALIZED_INPUT_CLAMP_MAX = 1.0
NORMALIZED_INPUT_ERASING_VALUE = 0.0

# 2. FUNKCIE PRE DYNAMICKU SIMULACIU DAT
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

def wrap_phase(img):
    return (img + np.pi) % (2 * np.pi) - np.pi

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
    a_val *= amplify_ab; b_val *= amplify_ab
    poly1d = a_val * x_norm**3 + b_val * x_norm**2 + c_val * x_norm + d_val
    poly2d = np.tile(poly1d, (H, 1))
    background = linear_grad + poly2d
    return background, dict(a=a_val, b=b_val, c=c_val, d=d_val, n_strips_actual=n_strips,
                            slope_y=slope_y, slope_x=slope_x, tilt_angle_deg=tilt_angle_deg)

def generate_simulation_pair_from_source_np_for_training(
    source_image_np, param_ranges_config, amplify_ab_value
):
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
    if smooth_original_image_sigma > 0 and original_image_influence > 0:
        img_phase_obj_base = gaussian(source_image_np, sigma=smooth_original_image_sigma, preserve_range=True, channel_axis=None)
    else:
        img_phase_obj_base = source_image_np
    object_phase_contribution = img_phase_obj_base * (2 * np.pi)
    generated_background, _ = generate_cubic_background(
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

# 3. POMOCNE FUNKCIE PRE MANIPULACIU S DATAMI (Normalizacia, Statistiky)
def denormalize_input_minmax_from_minus_one_one(data_norm_minus_one_one, original_min, original_max):
    if not isinstance(data_norm_minus_one_one, torch.Tensor):
        data_norm_minus_one_one = torch.tensor(data_norm_minus_one_one)
    if original_max == original_min:
        return torch.full_like(data_norm_minus_one_one, original_min)
    return (data_norm_minus_one_one + 1.0) * (original_max - original_min) / 2.0 + original_min

def calculate_wrapped_input_min_max_from_ref_dataset(dataset_path, data_type_name="Wrapped Input Data (from Ref)"):
    image_files = glob.glob(os.path.join(dataset_path, 'images', "*.tiff"))
    if not image_files:
        raise FileNotFoundError(f"Nenasli sa ziadne obrazky v {os.path.join(dataset_path, 'images')} pre vypocet Min/Max pre {data_type_name}.")

    min_val_g, max_val_g = np.inf, -np.inf
    print(f"Pocitam Min/Max pre {data_type_name} z {len(image_files)} suborov v {dataset_path}...")
    for i, fp in enumerate(image_files):
        try:
            img = tiff.imread(fp).astype(np.float32)
            min_val_g = min(min_val_g, img.min())
            max_val_g = max(max_val_g, img.max())
            if (i + 1) % 100 == 0 or (i + 1) == len(image_files):
                print(f"  Spracovanych {i + 1}/{len(image_files)}...")
        except Exception as e:
            print(f"Chyba pri spracovani suboru {fp} pre Min/Max: {e}")

    if np.isinf(min_val_g) or np.isinf(max_val_g):
        raise ValueError(f"Nepodarilo sa urcit platne Min/Max hodnoty pre {data_type_name} z {dataset_path}.")
    print(f"Vypocet Min/Max pre {data_type_name} dokonceny: Min={min_val_g:.4f}, Max={max_val_g:.4f}")
    return min_val_g, max_val_g

def calculate_k_max_from_ref_dataset(dataset_path, target_img_size_for_consistency_check=None):
    images_dir = os.path.join(dataset_path, 'images')
    labels_dir = os.path.join(dataset_path, 'labels')

    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        raise FileNotFoundError(f"Adresare 'images' alebo 'labels' neboli najdene v {dataset_path}")

    image_files = sorted(glob.glob(os.path.join(images_dir, "*.tiff")))
    if not image_files:
        raise FileNotFoundError(f"Nenasli sa ziadne obrazky v {images_dir} pre vypocet k_max.")

    max_abs_k_found = 0.0
    print(f"Pocitam k_max z {len(image_files)} parov v {dataset_path}...")

    for i, img_path in enumerate(image_files):
        base_id_name = os.path.basename(img_path).replace('wrappedbg_', '').replace('.tiff', '')
        lbl_path = os.path.join(labels_dir, f'unwrapped_{base_id_name}.tiff')

        if not os.path.exists(lbl_path):
            print(f"VAROVANIE: Chybajuci label {lbl_path} pre obrazok {img_path}. Preskakujem.")
            continue

        try:
            wrapped_orig_np = tiff.imread(img_path).astype(np.float32)
            unwrapped_orig_np = tiff.imread(lbl_path).astype(np.float32)

            if wrapped_orig_np.shape != unwrapped_orig_np.shape:
                print(f"VAROVANIE: Rozdielne tvary pre {img_path} ({wrapped_orig_np.shape}) a {lbl_path} ({unwrapped_orig_np.shape}). Preskakujem.")
                continue

            k_diff_np = (unwrapped_orig_np - wrapped_orig_np) / (2 * np.pi)
            k_float_np = np.round(k_diff_np)

            current_max_abs_k = np.max(np.abs(k_float_np))
            if current_max_abs_k > max_abs_k_found:
                max_abs_k_found = current_max_abs_k

            if (i + 1) % 100 == 0 or (i + 1) == len(image_files):
                print(f"  Spracovanych {i + 1}/{len(image_files)} pre k_max... Aktualne max_abs_k: {max_abs_k_found:.2f}")

        except Exception as e:
            print(f"Chyba pri spracovani paru {img_path}/{lbl_path} pre k_max: {e}")

    calculated_k_max = int(np.ceil(max_abs_k_found))
    print(f"Vypocet k_max dokonceny. Najdene max_abs_k_float: {max_abs_k_found:.2f}, Vypocitane k_max_val (ceil): {calculated_k_max}")
    return calculated_k_max

# 4. AUGMENTACNA TRANSFORMACIA
class AddGaussianNoiseTransform(nn.Module):
    def __init__(self, std_dev_range=(0.03, 0.12), p=0.5, clamp_min=None, clamp_max=None):
        super().__init__()
        self.std_dev_min, self.std_dev_max = std_dev_range
        self.p = p
        self.clamp_min, self.clamp_max = clamp_min, clamp_max
    def forward(self, img_tensor):
        if torch.rand(1).item() < self.p:
            std_dev = torch.empty(1).uniform_(self.std_dev_min, self.std_dev_max).item()
            noise = torch.randn_like(img_tensor) * std_dev
            noisy_img = img_tensor + noise
            if self.clamp_min is not None and self.clamp_max is not None:
                noisy_img = torch.clamp(noisy_img, self.clamp_min, self.clamp_max)
            return noisy_img
        return img_tensor

# 5. FUNKCIA PRE KONTROLU INTEGRITY DATASETU
def check_dataset_integrity(dataset_path):
    images_dir = os.path.join(dataset_path, 'images')
    labels_dir = os.path.join(dataset_path, 'labels')
    if not os.path.exists(images_dir): print(f"VAROVANIE: Adresar s obrazkami {images_dir} nebol najdeny."); return
    if not os.path.exists(labels_dir): print(f"VAROVANIE: Adresar s labelmi {labels_dir} nebol najdeny."); return

    image_files = sorted(glob.glob(os.path.join(images_dir, "*.tiff")))
    if not image_files: print(f"VAROVANIE: Nenasli sa ziadne TIFF obrazky v {images_dir}."); return

    missing_labels = 0
    for image_file in image_files:
        label_file_name = os.path.basename(image_file).replace('wrappedbg', 'unwrapped')
        expected_label_path = os.path.join(labels_dir, label_file_name)
        if not os.path.exists(expected_label_path):
            missing_labels +=1
            print(f"  Chybajuci label: {expected_label_path} pre obrazok {image_file}")
    if missing_labels == 0: print(f"Dataset {dataset_path} je v poriadku (kontrola nazvov suborov).")
    else: print(f"Dataset {dataset_path} ma {missing_labels} chybajucich labelov.")

# 6. DATASETY (Staticky a Dynamicky)
class WrapCountDataset(Dataset):
    def __init__(self, path_to_data,
                 input_min_max_global,
                 k_max_val=KMAX_DEFAULT,
                 target_img_size=(512,512),
                 edge_loss_weight=3.0):
        self.path = path_to_data
        self.image_list = sorted(glob.glob(os.path.join(self.path, 'images', "*.tiff")))
        self.input_min_g, self.input_max_g = input_min_max_global
        self.k_max = k_max_val
        self.target_img_size = target_img_size
        self.edge_loss_weight = edge_loss_weight

        self.geometric_transforms = None
        self.pixel_transforms_for_input = None

    def _normalize_input_minmax_to_minus_one_one(self, data, min_val, max_val):
        if max_val == min_val: return torch.zeros_like(data)
        return 2.0 * (data - min_val) / (max_val - min_val) - 1.0

    def _ensure_shape_and_type(self, img_numpy, target_shape, data_name="Image", dtype=np.float32):
        img_numpy = img_numpy.astype(dtype)
        current_shape = img_numpy.shape[-2:]
        if current_shape != target_shape:
            h, w = current_shape; target_h, target_w = target_shape
            pad_h = max(0, target_h - h); pad_w = max(0, target_w - w)
            if pad_h > 0 or pad_w > 0:
                pad_top, pad_bottom = pad_h // 2, pad_h - (pad_h // 2)
                pad_left, pad_right = pad_w // 2, pad_w - (pad_w // 2)
                if img_numpy.ndim == 2: img_numpy = np.pad(img_numpy, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='reflect')
                elif img_numpy.ndim == 3 and img_numpy.shape[0] == 1: img_numpy = np.pad(img_numpy, ((0,0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='reflect')
                else: raise ValueError(f"Nepodporovany tvar pre padding: {img_numpy.shape}")
            h, w = img_numpy.shape[-2:]
            if h > target_h or w > target_w:
                start_h, start_w = (h - target_h) // 2, (w - target_w) // 2
                if img_numpy.ndim == 2: img_numpy = img_numpy[start_h:start_h+target_h, start_w:start_w+target_w]
                elif img_numpy.ndim == 3 and img_numpy.shape[0] == 1: img_numpy = img_numpy[:, start_h:start_h+target_h, start_w:start_w+target_w]
                else: raise ValueError(f"Nepodporovany tvar pre cropping: {img_numpy.shape}")
            if img_numpy.shape[-2:] != target_shape:
                  raise ValueError(f"{data_name} '{getattr(self, 'current_img_path_for_debug', 'N/A')}' po uprave tvaru {img_numpy.shape[-2:]}, ocakava sa {target_shape}")
        return img_numpy

    def __len__(self): return len(self.image_list)

    def __getitem__(self, index):
        self.current_img_path_for_debug = self.image_list[index]
        img_path = self.image_list[index]
        base_id_name = os.path.basename(img_path).replace('wrappedbg_', '').replace('.tiff','')
        lbl_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'labels', f'unwrapped_{base_id_name}.tiff')

        try:
            wrapped_orig_np = tiff.imread(img_path)
            unwrapped_orig_np = tiff.imread(lbl_path)
        except Exception as e:
            print(f"CHYBA nacitania statickeho paru: {img_path} alebo {lbl_path}. Error: {e}")
            return None,None,None,None,None

        wrapped_orig_np = self._ensure_shape_and_type(wrapped_orig_np, self.target_img_size, "Wrapped phase (static)")
        unwrapped_orig_np = self._ensure_shape_and_type(unwrapped_orig_np, self.target_img_size, "Unwrapped phase (static)")

        wrapped_tensor = torch.from_numpy(wrapped_orig_np.copy()).unsqueeze(0)
        unwrapped_tensor = torch.from_numpy(unwrapped_orig_np.copy()).unsqueeze(0)

        diff = (unwrapped_tensor - wrapped_tensor) / (2 * np.pi)
        k_float = torch.round(diff)
        k_float = torch.clamp(k_float, -self.k_max, self.k_max)
        k_label = (k_float + self.k_max).long().squeeze(0)

        wrapped_input_norm = self._normalize_input_minmax_to_minus_one_one(wrapped_tensor.squeeze(0), self.input_min_g, self.input_max_g).unsqueeze(0)

        # Generovanie weight map
        k_label_np = k_label.cpu().numpy()
        edge_mask = np.zeros_like(k_label_np, dtype=bool)
        # Horizontálne rozdiely
        edge_mask[:, :-1] |= (k_label_np[:, :-1] != k_label_np[:, 1:])
        edge_mask[:, 1:] |= (k_label_np[:, 1:] != k_label_np[:, :-1])
        # Vertikálne rozdiely
        edge_mask[:-1, :] |= (k_label_np[:-1, :] != k_label_np[1:, :])
        edge_mask[1:, :] |= (k_label_np[1:, :] != k_label_np[:-1, :])
        weight_map_np = np.ones_like(k_label_np, dtype=np.float32)
        weight_map_np[edge_mask] = self.edge_loss_weight # Použitie uloženej váhy
        weight_map_tensor = torch.from_numpy(weight_map_np)

        return wrapped_input_norm, k_label, unwrapped_tensor.squeeze(0), wrapped_tensor.squeeze(0), weight_map_tensor

class DynamicTrainWrapCountDataset(Dataset):
    def __init__(self,
                 source_image_filepaths,
                 simulation_param_ranges,
                 amplify_ab_fixed_value,
                 input_min_max_global,
                 k_max_val=KMAX_DEFAULT,
                 augmentation_strength='none',
                 target_img_size=(512,512),
                 edge_loss_weight=3.0):

        self.source_image_filepaths = source_image_filepaths
        self.simulation_param_ranges = simulation_param_ranges
        self.amplify_ab_fixed_value = amplify_ab_fixed_value
        self.input_min_g, self.input_max_g = input_min_max_global
        self.k_max = k_max_val
        self.augmentation_strength = augmentation_strength
        self.target_img_size = target_img_size
        self.edge_loss_weight = edge_loss_weight # Uloženie váhy

        self.geometric_transforms = None
        self.pixel_transforms_for_input = None
        if self.augmentation_strength != 'none':
            self._setup_augmentations(self.augmentation_strength)

    def _setup_augmentations(self, strength):
        noise_std_range, noise_p = (0.01, 0.05), 0.0
        erase_scale, erase_p = (0.01, 0.04), 0.0

        if strength == 'light':
            noise_std_range, noise_p = (0.02, 0.08), 0.4 
            erase_scale, erase_p = (0.01, 0.05), 0.3
        elif strength == 'medium':
            p_affine = 0.5
            noise_std_range, noise_p = (0.03, 0.12), 0.5
            erase_scale, erase_p = (0.02, 0.08), 0.4
        elif strength == 'strong':
            p_affine = 0.6
            noise_std_range, noise_p = (0.05, 0.15), 0.6
            erase_scale, erase_p = (0.02, 0.10), 0.5

        geo_transforms_list = []
        if strength in ['light', 'medium', 'strong']: 
             geo_transforms_list.append(T.RandomHorizontalFlip(p=0.5))
        if geo_transforms_list:
            self.geometric_transforms = T.Compose(geo_transforms_list)

        pixel_aug_list = []
        if noise_p > 0:
            pixel_aug_list.append(AddGaussianNoiseTransform(std_dev_range=noise_std_range, p=noise_p,
                                          clamp_min=NORMALIZED_INPUT_CLAMP_MIN,
                                          clamp_max=NORMALIZED_INPUT_CLAMP_MAX))
        if erase_p > 0:
            pixel_aug_list.append(T.RandomErasing(p=erase_p, scale=erase_scale, ratio=(0.3, 3.3),
                                value=NORMALIZED_INPUT_ERASING_VALUE, inplace=False))
        if pixel_aug_list:
            self.pixel_transforms_for_input = T.Compose(pixel_aug_list)


    def _normalize_input_minmax_to_minus_one_one(self, data, min_val, max_val):
        if max_val == min_val: return torch.zeros_like(data)
        return 2.0 * (data - min_val) / (max_val - min_val) - 1.0

    def __len__(self):
        return len(self.source_image_filepaths)

    def __getitem__(self, idx):
        source_img_path = self.source_image_filepaths[idx % len(self.source_image_filepaths)]
        try:
            source_img_raw = io.imread(source_img_path).astype(np.float32)
            if source_img_raw.shape != self.target_img_size:
                 print(f"VAROVANIE: Zdrojovy obrazok {source_img_path} ma tvar {source_img_raw.shape}, ocakava sa {self.target_img_size}. Skusim resize.")
                 source_img_raw_tensor = torch.from_numpy(source_img_raw).unsqueeze(0).unsqueeze(0)
                 source_img_raw_tensor = F.interpolate(source_img_raw_tensor, size=self.target_img_size, mode='bilinear', align_corners=False)
                 source_img_raw = source_img_raw_tensor.squeeze(0).squeeze(0).numpy()
        except Exception as e:
            print(f"CHYBA nacitania zdrojoveho obrazka: {source_img_path}. Error: {e}");
            return None, None, None, None, None # Pridaná piata None hodnota

        img_min_val, img_max_val = source_img_raw.min(), source_img_raw.max()
        source_img_norm_for_sim = (source_img_raw - img_min_val) / (img_max_val - img_min_val) if img_max_val > img_min_val else np.zeros_like(source_img_raw)

        unwrapped_phase_np, wrapped_phase_np = generate_simulation_pair_from_source_np_for_training(
            source_img_norm_for_sim, self.simulation_param_ranges, self.amplify_ab_fixed_value
        )

        wrapped_tensor_orig_geo_aug = torch.from_numpy(wrapped_phase_np.copy()).unsqueeze(0)
        unwrapped_tensor_orig_geo_aug = torch.from_numpy(unwrapped_phase_np.copy()).unsqueeze(0)

        if self.geometric_transforms:
            stacked_phases = torch.cat((wrapped_tensor_orig_geo_aug, unwrapped_tensor_orig_geo_aug), dim=0)
            stacked_phases_aug = self.geometric_transforms(stacked_phases)
            wrapped_tensor_orig_geo_aug, unwrapped_tensor_orig_geo_aug = stacked_phases_aug[0:1], stacked_phases_aug[1:2]


        diff = (unwrapped_tensor_orig_geo_aug - wrapped_tensor_orig_geo_aug) / (2 * np.pi)
        k_float = torch.round(diff)
        k_float = torch.clamp(k_float, -self.k_max, self.k_max)
        k_label = (k_float + self.k_max).long().squeeze(0)

        wrapped_input_norm = self._normalize_input_minmax_to_minus_one_one(
            wrapped_tensor_orig_geo_aug.squeeze(0), self.input_min_g, self.input_max_g
        ).unsqueeze(0)

        if self.pixel_transforms_for_input:
            wrapped_input_norm = self.pixel_transforms_for_input(wrapped_input_norm)

        # Generovanie weight map z k_label
        k_label_np = k_label.cpu().numpy()
        edge_mask = np.zeros_like(k_label_np, dtype=bool)
        edge_mask[:, :-1] |= (k_label_np[:, :-1] != k_label_np[:, 1:])
        edge_mask[:, 1:] |= (k_label_np[:, 1:] != k_label_np[:, :-1])
        edge_mask[:-1, :] |= (k_label_np[:-1, :] != k_label_np[1:, :])
        edge_mask[1:, :] |= (k_label_np[1:, :] != k_label_np[:-1, :])

        weight_map_np = np.ones_like(k_label_np, dtype=np.float32)
        weight_map_np[edge_mask] = self.edge_loss_weight
        weight_map_tensor = torch.from_numpy(weight_map_np)

        return wrapped_input_norm, k_label, unwrapped_tensor_orig_geo_aug.squeeze(0), wrapped_tensor_orig_geo_aug.squeeze(0), weight_map_tensor

# 7. COLLATE FUNKCIA PRE DATALOADER
def collate_fn_skip_none_classification(batch):
    batch = list(filter(lambda x: all(item is not None for item in x[:5]), batch))
    if not batch: return None, None, None, None, None
    return torch.utils.data.dataloader.default_collate(batch)

# 8. VYPOCTOVE FUNKCIE (Loss a Metriky)
def cross_entropy_loss_full(logits, klabels, weight_map=None):
    pixel_losses = F.cross_entropy(logits, klabels, reduction='none') 
    if weight_map is not None:
        weighted_losses = pixel_losses * weight_map
        return torch.mean(weighted_losses)
    else:
        return torch.mean(pixel_losses)

def k_label_accuracy_full(logits, klabels):
    pred_classes = torch.argmax(logits, dim=1)
    correct = (pred_classes == klabels).float().sum()
    total = klabels.numel()
    return correct / total

# 9. FUNKCIA PRE TRENOVACIU SESSIU KLASIFIKACIE
def run_classification_training_session(
    run_id, device, num_epochs, train_loader, val_loader, test_loader,
    input_min_max_for_denorm, 
    encoder_name, encoder_weights, k_max_val,
    learning_rate, weight_decay, 
    cosine_T_max, cosine_eta_min,
    early_stopping_patience,
    augmentation_strength, 
    train_data_source_type,
    edge_loss_weight_value 
    ):

    config_save_path = f'config_clf_{run_id}.txt'
    num_classes_effective = 2 * k_max_val + 1
    config_details = {
        "Run ID": run_id, "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Task Type": "Classification (Wrap Count) + Reconstruction MAE",
        "Encoder Name": encoder_name, "Encoder Weights": encoder_weights,
        "K_MAX": k_max_val, "NUM_CLASSES": num_classes_effective,
        "Input Normalization (Global MinMax for Wrapped)": f"Min: {input_min_max_for_denorm[0]:.4f}, Max: {input_min_max_for_denorm[1]:.4f}",
        "Train Data Source": train_data_source_type,
        "Train Augmentation Strength": augmentation_strength,
        "Edge Loss Weight": edge_loss_weight_value, 
        "Initial LR": learning_rate, "Batch Size": train_loader.batch_size if train_loader else "N/A", 
        "Num Epochs": num_epochs, "Weight Decay": weight_decay,
        "Scheduler Type": "CosineAnnealingLR", 
        "CosineAnnealingLR T_max": cosine_T_max, 
        "CosineAnnealingLR eta_min": cosine_eta_min, 
        "EarlyStopping Patience": early_stopping_patience, "Device": str(device),
    }
    with open(config_save_path, 'w') as f:
        f.write("Experiment Configuration:\n" + "="*25 + "\n" + 
                "\n".join([f"{k}: {v}" for k,v in config_details.items()]) + "\n")
    print(f"Konfiguracia klasifikacneho experimentu ulozena do: {config_save_path}")

    net = smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights,
                   in_channels=1, classes=num_classes_effective, activation=None).to(device)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_T_max, eta_min=cosine_eta_min)

    train_ce_loss_hist, val_ce_loss_hist = [], []
    train_k_acc_hist, val_k_acc_hist = [], []
    val_mae_hist = [] 

    best_val_k_accuracy = 0.0
    best_val_mae = float('inf')
    weights_path = f'best_weights_clf_{run_id}.pth'
    
    epochs_no_k_acc_improve = 0
    epochs_no_mae_improve = 0
    
    print(f"Starting CLASSIFICATION training for {run_id}...")

    epoch_viz_main_dir = f"epoch_visualizations_clf_{run_id}"
    os.makedirs(epoch_viz_main_dir, exist_ok=True)

    for epoch in range(num_epochs):
        start_time = time.time()
        net.train()
        epoch_train_ce, epoch_train_k_acc = [], []
        for batch_data in train_loader:
            # batch_data teraz obsahuje 5 položiek
            if batch_data[0] is None: continue 
            wrapped_norm, k_labels_gt, _, _, weight_maps = batch_data
            
            wrapped_norm, k_labels_gt = wrapped_norm.to(device), k_labels_gt.to(device)
            weight_maps = weight_maps.to(device)

            optimizer.zero_grad()
            logits = net(wrapped_norm)
            loss = cross_entropy_loss_full(logits, k_labels_gt, weight_maps)
            loss.backward(); optimizer.step()
            
            with torch.no_grad():
                acc = k_label_accuracy_full(logits, k_labels_gt)
            epoch_train_ce.append(loss.item())
            epoch_train_k_acc.append(acc.item())
        
        avg_train_ce = np.mean(epoch_train_ce) if epoch_train_ce else float('nan')
        avg_train_k_acc = np.mean(epoch_train_k_acc) if epoch_train_k_acc else float('nan')
        train_ce_loss_hist.append(avg_train_ce)
        train_k_acc_hist.append(avg_train_k_acc)

        net.eval()
        epoch_val_ce, epoch_val_k_acc, epoch_val_mae = [],[],[] 
        visualized_this_epoch = False 
        with torch.no_grad():
            for val_batch_idx, batch_data_val in enumerate(val_loader): 
                if batch_data_val[0] is None: continue
                
                wrapped_norm_val, k_labels_gt_val, unwrapped_gt_orig_val, wrapped_orig_val, weight_maps_val = batch_data_val
                
                wrapped_norm_val = wrapped_norm_val.to(device)
                k_labels_gt_val = k_labels_gt_val.to(device)
                weight_maps_val = weight_maps_val.to(device)
                unwrapped_gt_orig_val = unwrapped_gt_orig_val.to(device) 
                wrapped_orig_val = wrapped_orig_val.to(device) 


                logits_val = net(wrapped_norm_val)
                
                epoch_val_ce.append(cross_entropy_loss_full(logits_val, k_labels_gt_val, weight_maps_val).item())
                epoch_val_k_acc.append(k_label_accuracy_full(logits_val, k_labels_gt_val).item())

                pred_classes_val = torch.argmax(logits_val, dim=1) 
                k_pred_values_val = pred_classes_val.float() - k_max_val 
                unwrapped_pred_reconstructed_val = wrapped_orig_val + (2 * np.pi) * k_pred_values_val
                mae_denorm_val = torch.mean(torch.abs(unwrapped_pred_reconstructed_val - unwrapped_gt_orig_val))
                epoch_val_mae.append(mae_denorm_val.item())
                
                if val_batch_idx == 0 and not visualized_this_epoch and len(wrapped_orig_val) > 0:
                    pred_classes_val_viz = torch.argmax(logits_val, dim=1) 
                    j = 0 
                    wrapped_display_val = wrapped_orig_val[j].cpu().numpy().squeeze()
                    k_labels_gt_display_val = k_labels_gt_val[j].cpu().numpy().squeeze()
                    pred_classes_display_val = pred_classes_val_viz[j].cpu().numpy().squeeze()

                    fig_epoch_k, axes_epoch_k = plt.subplots(1, 3, figsize=(18, 6))
                    fig_epoch_k.suptitle(f"Val. k-triedy - {run_id} - Epocha {epoch+1}", fontsize=16)
                    
                    im0_val = axes_epoch_k[0].imshow(wrapped_display_val, cmap='gray')
                    axes_epoch_k[0].set_title(f"Vstup Wrapped (orig.)", fontsize=14)
                    fig_epoch_k.colorbar(im0_val, ax=axes_epoch_k[0])
                    
                    im1_val = axes_epoch_k[1].imshow(k_labels_gt_display_val, cmap='viridis', vmin=0, vmax=num_classes_effective-1)
                    axes_epoch_k[1].set_title(f"GT k-triedy (0 až {num_classes_effective-1})", fontsize=14)
                    fig_epoch_k.colorbar(im1_val, ax=axes_epoch_k[1])
                    
                    im2_val = axes_epoch_k[2].imshow(pred_classes_display_val, cmap='viridis', vmin=0, vmax=num_classes_effective-1)
                    axes_epoch_k[2].set_title(f"Predikované k-triedy (0 až {num_classes_effective-1})", fontsize=14)
                    fig_epoch_k.colorbar(im2_val, ax=axes_epoch_k[2])
                    
                    plt.tight_layout(rect=[0,0,1,0.95])
                    plt.savefig(os.path.join(epoch_viz_main_dir, f'val_k_labels_ep{epoch+1}.png'))
                    plt.close(fig_epoch_k)
                    visualized_this_epoch = True 
        
        avg_val_ce = np.mean(epoch_val_ce) if epoch_val_ce else float('nan')
        avg_val_k_acc = np.mean(epoch_val_k_acc) if epoch_val_k_acc else float('nan')
        avg_val_mae = np.mean(epoch_val_mae) if epoch_val_mae else float('nan') 
        
        val_ce_loss_hist.append(avg_val_ce)
        val_k_acc_hist.append(avg_val_k_acc)
        val_mae_hist.append(avg_val_mae) 

        epoch_duration = time.time() - start_time
        print(f"Run: {run_id} | Ep {epoch+1}/{num_epochs} | Tr CE: {avg_train_ce:.4f}, Tr kAcc: {avg_train_k_acc:.4f} | Val CE: {avg_val_ce:.4f}, Val kAcc: {avg_val_k_acc:.4f}, Val MAE: {avg_val_mae:.4f} | LR: {optimizer.param_groups[0]['lr']:.1e} | Time: {epoch_duration:.2f}s")

        # Early stopping logika
        k_acc_improved = False
        mae_improved = False

        if not np.isnan(avg_val_k_acc) and avg_val_k_acc > best_val_k_accuracy:
            best_val_k_accuracy = avg_val_k_acc
            torch.save(net.state_dict(), weights_path)
            print(f"  New best Val k-Acc: {best_val_k_accuracy:.4f} (Val MAE: {avg_val_mae:.4f}). Saved model based on k-Acc.")
            epochs_no_k_acc_improve = 0
            k_acc_improved = True
        elif not np.isnan(avg_val_k_acc):
            epochs_no_k_acc_improve += 1
        
        if not np.isnan(avg_val_mae) and avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            # Model sa neukladá na základe MAE, len sa sleduje najlepšia hodnota
            print(f"  New best Val MAE: {best_val_mae:.4f} (Val k-Acc: {avg_val_k_acc:.4f}).")
            epochs_no_mae_improve = 0
            mae_improved = True
        elif not np.isnan(avg_val_mae):
            epochs_no_mae_improve += 1
        
        # Zastaví tréning, ak sa ani k-Acc ani MAE nezlepšili počas 'patience' epoch
        if epochs_no_k_acc_improve >= early_stopping_patience and \
           epochs_no_mae_improve >= early_stopping_patience:
            print(f"Early stopping @ epoch {epoch+1}: Val k-Acc not improved for {epochs_no_k_acc_improve} epochs AND Val MAE not improved for {epochs_no_mae_improve} epochs.")
            break
        
        scheduler.step() 
    
    print(f"Training of {run_id} done. Best Val k-Acc (used for saving model): {best_val_k_accuracy:.4f}. Best Val MAE (tracked): {best_val_mae:.4f} @ {weights_path}")

    avg_test_mae_denorm = float('nan')
    avg_test_k_acc = float('nan')
    if os.path.exists(weights_path) and test_loader is not None:
        print(f"\nTesting with best weights for {run_id}...")
        net.load_state_dict(torch.load(weights_path, map_location=device))
        net.eval()
        test_mae_list_denorm = []
        test_k_acc_list = []
        
        with torch.no_grad():
            for i, batch_data_test in enumerate(test_loader):
                if batch_data_test[0] is None: continue
                wrapped_norm_test, k_labels_gt_test, unwrapped_gt_orig_test, wrapped_orig_test, _ = batch_data_test
                wrapped_norm_test = wrapped_norm_test.to(device)
                k_labels_gt_test = k_labels_gt_test.to(device)
                unwrapped_gt_orig_test = unwrapped_gt_orig_test.to(device)
                wrapped_orig_test = wrapped_orig_test.to(device)

                logits_test = net(wrapped_norm_test)
                pred_classes_test = torch.argmax(logits_test, dim=1)
                
                current_k_acc = k_label_accuracy_full(logits_test, k_labels_gt_test)
                test_k_acc_list.append(current_k_acc.item())

                k_pred_values_test = pred_classes_test.float() - k_max_val
                unwrapped_pred_reconstructed = wrapped_orig_test + (2 * np.pi) * k_pred_values_test

                mae_denorm = torch.mean(torch.abs(unwrapped_pred_reconstructed - unwrapped_gt_orig_test))
                test_mae_list_denorm.append(mae_denorm.item())

                if i == 0 and len(wrapped_norm_test) > 0:
                    j = 0
                    
                    # Pripravíme dáta pre vizualizáciu (CPU, numpy)
                    wrapped_display = wrapped_orig_test[j].cpu().numpy().squeeze()
                    k_labels_gt_display = k_labels_gt_test[j].cpu().numpy().squeeze()
                    pred_classes_display = pred_classes_test[j].cpu().numpy().squeeze()
                    unwrapped_gt_display = unwrapped_gt_orig_test[j].cpu().numpy().squeeze()
                    unwrapped_reconstructed_display = unwrapped_pred_reconstructed[j].cpu().numpy().squeeze()

                    # Vizualizácia 1: k-triedy
                    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))
                    fig1.suptitle(f"Vizualizácia k-tried - {run_id}", fontsize=16)
                    
                    im0 = axes1[0].imshow(wrapped_display, cmap='gray')
                    axes1[0].set_title(f"Vstup Wrapped (denorm/orig.)", fontsize=14)
                    fig1.colorbar(im0, ax=axes1[0])
                    
                    im1 = axes1[1].imshow(k_labels_gt_display, cmap='viridis', vmin=0, vmax=num_classes_effective-1)
                    axes1[1].set_title(f"GT k-triedy (0 až {num_classes_effective-1})", fontsize=14)
                    fig1.colorbar(im1, ax=axes1[1])
                    
                    im2 = axes1[2].imshow(pred_classes_display, cmap='viridis', vmin=0, vmax=num_classes_effective-1)
                    axes1[2].set_title(f"Predikované k-triedy (0 až {num_classes_effective-1})", fontsize=14)
                    fig1.colorbar(im2, ax=axes1[2])
                    
                    plt.tight_layout(rect=[0,0,1,0.95])
                    plt.savefig(f'vis_clf_k_labels_{run_id}.png')
                    plt.close(fig1)

                    # Vizualizácia 2: Rekonštrukcia
                    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
                    fig2.suptitle(f"Vizualizácia rekonštrukcie fázy - {run_id}", fontsize=16)

                    im3 = axes2[0].imshow(wrapped_display, cmap='gray')
                    axes2[0].set_title(f"Vstup Wrapped (denorm/orig.)", fontsize=14)
                    fig2.colorbar(im3, ax=axes2[0])

                    im4 = axes2[1].imshow(unwrapped_gt_display, cmap='gray')
                    axes2[1].set_title(f"GT Unwrapped Fáza", fontsize=14)
                    fig2.colorbar(im4, ax=axes2[1])

                    im5 = axes2[2].imshow(unwrapped_reconstructed_display, cmap='gray')
                    axes2[2].set_title(f"Rekonštruovaná Unwrapped Fáza", fontsize=14)
                    fig2.colorbar(im5, ax=axes2[2])

                    plt.tight_layout(rect=[0,0,1,0.95])
                    plt.savefig(f'vis_clf_reconstruction_{run_id}.png')
                    plt.close(fig2)

                    try:
                        tiff.imwrite(f'input_wrapped_orig_test_{run_id}.tiff', wrapped_display.astype(np.float32))
                        tiff.imwrite(f'gt_k_labels_test_{run_id}.tiff', k_labels_gt_display.astype(np.uint8))
                        tiff.imwrite(f'pred_k_labels_test_{run_id}.tiff', pred_classes_display.astype(np.uint8))
                        tiff.imwrite(f'gt_unwrapped_test_{run_id}.tiff', unwrapped_gt_display.astype(np.float32))
                        tiff.imwrite(f'reconstructed_unwrapped_test_{run_id}.tiff', unwrapped_reconstructed_display.astype(np.float32))
                    except Exception as e_tiff:
                        print(f"Chyba pri ukladaní testovacích TIFF obrázkov: {e_tiff}")

        avg_test_mae_denorm = np.mean(test_mae_list_denorm) if test_mae_list_denorm else float('nan')
        avg_test_k_acc = np.mean(test_k_acc_list) if test_k_acc_list else float('nan')
        
        print(f"  Test k-label Accuracy: {avg_test_k_acc:.4f}")
        print(f"  Test MAE (Denorm, Rekonštrukcia): {avg_test_mae_denorm:.6f}")
        
        with open(f"metrics_clf_{run_id}.txt", "w") as f:
            f.write(f"Run ID: {run_id}\n")
            f.write(f"Best Val k-Accuracy (model saved): {best_val_k_accuracy:.4f}\n")
            f.write(f"Best Val MAE (tracked during training): {best_val_mae:.4f}\n") 
            f.write(f"Test k-Accuracy: {avg_test_k_acc:.4f}\n")
            f.write(f"Test MAE (Denorm, Rekonštrukcia): {avg_test_mae_denorm:.6f}\n")
    else:
        if not os.path.exists(weights_path): print(f"No weights found for {run_id} to test.")
        if test_loader is None: print(f"Test loader not provided for {run_id}, skipping testing.")

    plt.figure(figsize=(18,5)) 
    plt.subplot(1,3,1) 
    plt.plot(train_ce_loss_hist,label='Tréningová CE Loss')
    plt.plot(val_ce_loss_hist,label='Validačná CE Loss')
    plt.title(f'Priebeh CE Loss - {run_id}', fontsize=16); plt.xlabel('Epocha', fontsize=12); plt.ylabel('CE Loss', fontsize=12); plt.legend(); plt.grid(True)
    
    plt.subplot(1,3,2) 
    plt.plot(train_k_acc_hist,label='Treningova k-Accuracy')
    plt.plot(val_k_acc_hist,label='Validacna k-Accuracy')
    plt.title(f'Priebeh k-Label Accuracy - {run_id}', fontsize=16); plt.xlabel('Epocha', fontsize=12); plt.ylabel('Accuracy', fontsize=12); plt.legend(); plt.grid(True)

    plt.subplot(1,3,3) 
    plt.plot(val_mae_hist, label='Validacne MAE (Rekonstrukcia)')
    plt.title(f'Priebeh Validacneho MAE - {run_id}', fontsize=16); plt.xlabel('Epocha', fontsize=12); plt.ylabel('MAE', fontsize=12); plt.legend(); plt.grid(True)
    
    plt.tight_layout(); plt.savefig(f'curves_clf_{run_id}.png'); plt.close()

    return {"best_val_k_accuracy": best_val_k_accuracy, 
            "best_val_mae_overall": best_val_mae, 
            "test_mae_denorm": avg_test_mae_denorm, 
            "test_k_accuracy": avg_test_k_acc}

# 10. HLAVNA FUNKCIA PRE KONFIGURACIU A SPUSTENIE EXPERIMENTOV
def spusti_experimenty_klasifikacia():
    # --- CESTY K DATASETOM ---
    base_output_dir = "split_dataset_tiff_for_dynamic_v_stratified_final"
    path_static_ref_train = os.path.join(base_output_dir, "static_ref_train_dataset")
    path_valid_dataset = os.path.join(base_output_dir, 'static_valid_dataset')
    path_test_dataset = os.path.join(base_output_dir, 'static_test_dataset')
    path_dynamic_train_source_images = os.path.join(base_output_dir, "train_dataset_source_for_dynamic_generation", "images")

    # --- VÝPOČET NORMALIZAČNÝCH ŠTATISTÍK PRE WRAPPED VSTUP ---
    if not os.path.exists(os.path.join(path_static_ref_train, 'images')):
        raise FileNotFoundError(f"Adresar s obrazkami pre referencny treningovy dataset nebol najdeny: {os.path.join(path_static_ref_train, 'images')}")
    
    GLOBAL_WRAPPED_INPUT_MIN, GLOBAL_WRAPPED_INPUT_MAX = calculate_wrapped_input_min_max_from_ref_dataset(
        path_static_ref_train, "Global Wrapped Input (from Ref Train)"
    )
    norm_input_minmax_stats_global = (GLOBAL_WRAPPED_INPUT_MIN, GLOBAL_WRAPPED_INPUT_MAX)
    print(f"Pouzivaju sa vypocitane globalne statistiky pre wrapped vstup (normalizacia na [-1,1]): Min={GLOBAL_WRAPPED_INPUT_MIN:.4f}, Max={GLOBAL_WRAPPED_INPUT_MAX:.4f}")

    # --- VÝPOČET k_max_val Z REFERENČNÉHO DATASETU ---.
    CALCULATED_K_MAX_FROM_DATA = calculate_k_max_from_ref_dataset(path_static_ref_train)
    NUM_CLASSES_EFFECTIVE_FROM_DATA = 2 * CALCULATED_K_MAX_FROM_DATA + 1
    print(f"Globálne vypocítané k_max_val z referenčného datasetu: {CALCULATED_K_MAX_FROM_DATA}")
    print(f"Zodpovedajúci počet efektívnych tried: {NUM_CLASSES_EFFECTIVE_FROM_DATA}")


    if not os.path.exists(path_dynamic_train_source_images):
        raise FileNotFoundError(f"Adresar so zdrojovými obrázkami pre dynamický tréning nebol nájdený: {path_dynamic_train_source_images}")

    # Kontrola integrity pre statické sady
    for ds_path_check in [path_valid_dataset, path_test_dataset]:
        if os.path.exists(ds_path_check):
            check_dataset_integrity(ds_path_check)
        else:
            print(f"VAROVANIE: Adresár statického datasetu nebol nájdený: {ds_path_check}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_train_data_workers = 1
    num_eval_data_workers = 1

    # --- Parametre simulácie ---
    simulation_param_ranges_config = {
        "n_strips_param": (7, 8), 
        "original_image_influence_param": (0.3, 0.5),
        "phase_noise_std_param": (0.024, 0.039),
        "smooth_original_image_sigma_param": (0.2, 0.5),
        "poly_scale_param": (0.02, 0.1), 
        "CURVATURE_AMPLITUDE_param": (1.4, 2.0), 
        "background_offset_d_param": (-24.8, -6.8), 
        "tilt_angle_deg_param": (-5.0, 17.0) 
    }
    amplify_ab_fixed_config_val = 1.0 
    # ----------------------------------------------------
    
    experiments_clf = [
        # najlepsie nastavenia hyperparametrov
        {
            "run_id_suffix": "AUTO_GENERATED", 
            "encoder_name": "resnet34", "encoder_weights": "imagenet",
            "augmentation_strength": "medium", 
            "lr": 1e-3, "bs": 8, "epochs": 120, "wd": 1e-4,
            "es_pat": 30, 
            "cosine_T_max": 120, 
            "cosine_eta_min": 1e-7,
            "edge_loss_weight": 5.0 
        },
    ]

    all_clf_results = []
    for cfg_original in experiments_clf:
        cfg = cfg_original.copy() 
        
        # Globálne vypočítané k_max_val
        current_k_max_for_experiment = CALCULATED_K_MAX_FROM_DATA
        cfg['k_max_val'] = current_k_max_for_experiment

        # --- Dynamické generovanie run_id_suffix ---
        encoder_short_map = {"resnet18": "R18", "resnet34": "R34", "resnet50": "R50",
                             "efficientnet-b0": "EffB0", "efficientnet-b1": "EffB1"}
        enc_name_part = encoder_short_map.get(cfg["encoder_name"], cfg["encoder_name"][:5])
        enc_weights_part = "imgnet" if cfg["encoder_weights"] == "imagenet" else "scratch"
        enc_part = f"{enc_name_part}{enc_weights_part}"
        
        kmax_part = f"Kmax{current_k_max_for_experiment}"
        aug_part = f"Aug{cfg['augmentation_strength'][:3].capitalize()}" if cfg['augmentation_strength'] != 'none' else "AugNone"
        lr_part = f"LR{cfg['lr']:.0e}".replace('-', 'm')
        
        # Úprava pre bs_part pre robustnosť
        bs_value = cfg.get('bs') 
        bs_part = f"bs{bs_value}" if bs_value is not None else None

        wd_part = f"WD{cfg['wd']:.0e}".replace('-', 'm') if cfg.get('wd',0) > 0 else "WD0"
        epochs_part = f"Ep{cfg['epochs']}"
        
        # Časti pre CosineAnnealingLR scheduler
        cosine_T_max_part = f"Tmax{cfg['cosine_T_max']}"
        eta_min_val_cfg = cfg.get('cosine_eta_min')
        effective_eta_min_for_id = eta_min_val_cfg if eta_min_val_cfg is not None else cfg.get("min_lr", 1e-7)

        if effective_eta_min_for_id == 0:
            cosine_eta_min_part = "EtaMin0"
        else:
            cosine_eta_min_part = f"EtaMin{effective_eta_min_for_id:.0e}".replace('-', 'm')
        
        edge_weight_part = f"EdgeW{cfg.get('edge_loss_weight', 1.0)}"

        parts = [part for part in [enc_part, kmax_part, aug_part, lr_part, wd_part, epochs_part, cosine_T_max_part, cosine_eta_min_part, edge_weight_part, bs_part] if part]
        generated_suffix = "_".join(parts)

        if "run_id_suffix" in cfg_original and cfg_original["run_id_suffix"] != "AUTO_GENERATED":
            run_id_final = cfg_original["run_id_suffix"]
        else:
            run_id_final = generated_suffix
        cfg['run_id_final_used'] = run_id_final

        print(f"\n\n{'='*20} KLASIFIKACNY EXPERIMENT (DYN. TRENING): {run_id_final} {'='*20}")
        
        # --- Vytvorenie DataLoaders ---
        source_filepaths_for_train_loader = glob.glob(os.path.join(path_dynamic_train_source_images, "*.tif*"))
        if not source_filepaths_for_train_loader:
            print(f"CHYBA: Nenasli sa ziadne zdrojove obrazky v {path_dynamic_train_source_images}. Preskakujem experiment {run_id_final}.")
            all_clf_results.append({"run_id": run_id_final, "config": cfg, "metrics": {"error": "No source images for dynamic training"}})
            continue
            
        train_ds = DynamicTrainWrapCountDataset(
            source_image_filepaths=source_filepaths_for_train_loader,
            simulation_param_ranges=simulation_param_ranges_config,
            amplify_ab_fixed_value=amplify_ab_fixed_config_val,
            input_min_max_global=norm_input_minmax_stats_global,
            k_max_val=current_k_max_for_experiment, # Použijeme vypočítanú hodnotu
            augmentation_strength=cfg["augmentation_strength"],
            target_img_size=(512,512),
            edge_loss_weight=cfg.get('edge_loss_weight', 1.0) 
        )
        train_loader = DataLoader(train_ds, batch_size=cfg["bs"], shuffle=True, 
                                  num_workers=num_train_data_workers, 
                                  collate_fn=collate_fn_skip_none_classification, 
                                  pin_memory=device.type=='cuda', 
                                  persistent_workers=True if num_train_data_workers > 0 else False)
        
        val_loader = None
        if os.path.exists(path_valid_dataset) and len(glob.glob(os.path.join(path_valid_dataset, 'images', "*.tiff"))) > 0 :
            val_ds = WrapCountDataset(path_valid_dataset,
                                   input_min_max_global=norm_input_minmax_stats_global,
                                   k_max_val=current_k_max_for_experiment, # Použijeme vypočítanú hodnotu
                                   edge_loss_weight=cfg.get('edge_loss_weight', 1.0)) 
            val_loader = DataLoader(val_ds, batch_size=cfg["bs"], shuffle=False, 
                                    num_workers=num_eval_data_workers, 
                                    collate_fn=collate_fn_skip_none_classification, 
                                    pin_memory=device.type=='cuda',
                                    persistent_workers=True if num_eval_data_workers > 0 else False)
        else:
            print(f"VAROVANIE: Validacný dataset nebol nájdený alebo je prázdny v {path_valid_dataset}. Validácia bude preskočená.")

        test_loader = None
        if os.path.exists(path_test_dataset) and len(glob.glob(os.path.join(path_test_dataset, 'images', "*.tiff"))) > 0:
            test_ds = WrapCountDataset(path_test_dataset,
                                    input_min_max_global=norm_input_minmax_stats_global,
                                    k_max_val=current_k_max_for_experiment,
                                    edge_loss_weight=cfg.get('edge_loss_weight', 1.0)) 
            test_loader = DataLoader(test_ds, batch_size=cfg["bs"], shuffle=False, 
                                     num_workers=num_eval_data_workers, 
                                     collate_fn=collate_fn_skip_none_classification, 
                                     pin_memory=device.type=='cuda',
                                     persistent_workers=True if num_eval_data_workers > 0 else False)
        else:
            print(f"VAROVANIE: Testovací dataset nebol nájdený alebo je prázdny v {path_test_dataset}. Testovanie bude preskočené.")

        if val_loader is None:
            print(f"CHYBA: Validačný loader nie je dostupný pre {run_id_final}. Preskakujem tréning.")
            all_clf_results.append({"run_id": run_id_final, "config": cfg, "metrics": {"error": "Missing validation data"}})
            continue
        
        # Príprava cosine_eta_min pre volanie run_classification_training_session
        effective_cosine_eta_min = cfg.get('cosine_eta_min')
        if effective_cosine_eta_min is None:
            effective_cosine_eta_min = cfg.get('min_lr', 1e-7)

        exp_results = run_classification_training_session(
            run_id=run_id_final, device=device, num_epochs=cfg["epochs"],
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            input_min_max_for_denorm=norm_input_minmax_stats_global, 
            encoder_name=cfg["encoder_name"], encoder_weights=cfg["encoder_weights"],
            k_max_val=current_k_max_for_experiment,
            learning_rate=cfg["lr"], weight_decay=cfg.get("wd", 1e-5), 
            cosine_T_max=cfg["cosine_T_max"],
            cosine_eta_min=effective_cosine_eta_min, 
            early_stopping_patience=cfg["es_pat"], 
            augmentation_strength=cfg["augmentation_strength"],
            train_data_source_type="Dynamic Simulation",
            edge_loss_weight_value=cfg.get('edge_loss_weight', 1.0) 
        )
        all_clf_results.append({"run_id": run_id_final, "config": cfg, "metrics": exp_results})

    # ... (Súhrn výsledkov zostáva rovnaké) ...
    print("\n\n" + "="*30 + " SUHRN KLASIFIKACNYCH VYSLEDKOV " + "="*30)
    for summary in all_clf_results:
        metrics = summary.get('metrics', {})
        print(f"Run: {summary.get('run_id', 'N/A')}")
        print(f"  Best Val k-Acc (model saved): {metrics.get('best_val_k_accuracy', float('nan')):.4f}")
        print(f"  Best Val MAE (overall tracked): {metrics.get('best_val_mae_overall', float('nan')):.4f}") 
        print(f"  Test k-Acc:     {metrics.get('test_k_accuracy', float('nan')):.4f}")
        print(f"  Test MAE (D,Rek): {metrics.get('test_mae_denorm', float('nan')):.6f}")
        if "error" in metrics: print(f"  Chyba: {metrics['error']}")
        print("-" * 70)
    print(f"--- VSETKY KLASIFIKACNE EXPERIMENTY DOKONCENE ---")


# 11. VSTUPNY BOD SKRIPTU
if __name__ == '__main__':
    torch_seed = 42
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)
    spusti_experimenty_klasifikacia()
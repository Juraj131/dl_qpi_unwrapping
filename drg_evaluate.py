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
from tqdm import tqdm

# --- GLOBALNE KONSTANTY ---
NORMALIZED_INPUT_CLAMP_MIN = -1.0
NORMALIZED_INPUT_CLAMP_MAX = 1.0
NORMALIZED_INPUT_ERASING_VALUE = 0.0

# --- DEFINICIA DATASSETU ---
class CustomDataset(Dataset):
    def __init__(self, path_to_data,
                 input_processing_type,
                 norm_stats_input,
                 norm_stats_target,
                 augmentation_strength='none',
                 is_train_set=False,
                 target_img_size=(512,512)):
        # path_to_data: cesta k datum
        # input_processing_type: typ spracovania vstupu ('sincos' alebo 'direct_minmax')
        # norm_stats_input: (min, max) pre vstup (pre direct_minmax), alebo None
        # norm_stats_target: (mean, std) pre ciel
        # augmentation_strength: sila augmentacii (pre test 'none')
        # is_train_set: boolean, ci ide o treningovy set
        # target_img_size: cielova velkost obrazkov (H, W)
        self.path = path_to_data
        self.image_list = sorted(glob.glob(os.path.join(self.path, 'images', "*.tiff")))
        self.input_processing_type = input_processing_type
        self.input_min, self.input_max = (None, None)
        if input_processing_type == 'direct_minmax' and norm_stats_input:
            self.input_min, self.input_max = norm_stats_input
        elif input_processing_type == 'sincos' and norm_stats_input:
            print("Poznamka: norm_stats_input poskytnute pre sincos, ale zvycajne sa nepouzivaju na normalizaciu sin/cos.")

        self.target_mean, self.target_std = norm_stats_target
        self.is_train_set = is_train_set
        self.target_img_size = target_img_size
        self.augmentation_strength = augmentation_strength

        self.geometric_transforms = None
        self.pixel_transforms = None

    def _setup_augmentations(self, strength):
        # strength: sila augmentacii
        pass

    def _normalize_input_minmax_to_minus_one_one(self, data, min_val, max_val):
        # data: vstupne data (torch.Tensor)
        # min_val: minimalna hodnota pre normalizaciu
        # max_val: maximalna hodnota pre normalizaciu
        if max_val == min_val: return torch.zeros_like(data)
        return 2.0 * (data - min_val) / (max_val - min_val) - 1.0

    def _normalize_target_z_score(self, data, mean_val, std_val):
        # data: vstupne data (torch.Tensor)
        # mean_val: priemer pre normalizaciu
        # std_val: standardna odchylka pre normalizaciu
        if std_val < 1e-6: return data - mean_val
        return (data - mean_val) / std_val

    def _ensure_shape_and_type(self, img_numpy, target_shape, data_name="Image", dtype=np.float32):
        # img_numpy: obrazok ako numpy array
        # target_shape: cielovy tvar (H, W)
        # data_name: nazov dat pre logovanie
        # dtype: cielovy datovy typ
        img_numpy = img_numpy.astype(dtype)
        if img_numpy.shape[-2:] != target_shape:
            original_shape = img_numpy.shape
            h, w = img_numpy.shape[-2:]
            target_h, target_w = target_shape

            pad_h = max(0, target_h - h)
            pad_w = max(0, target_w - w)

            if pad_h > 0 or pad_w > 0:
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                if img_numpy.ndim == 2:
                    img_numpy = np.pad(img_numpy, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='reflect')
                elif img_numpy.ndim == 3:
                    img_numpy = np.pad(img_numpy, ((0,0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='reflect')

            h, w = img_numpy.shape[-2:]
            if h > target_h or w > target_w:
                start_h = (h - target_h) // 2
                start_w = (w - target_w) // 2
                if img_numpy.ndim == 2:
                    img_numpy = img_numpy[start_h:start_h+target_h, start_w:start_w+target_w]
                elif img_numpy.ndim == 3:
                    img_numpy = img_numpy[:, start_h:start_h+target_h, start_w:start_w+target_w]

            if img_numpy.shape[-2:] != target_shape:
                 print(f"VAROVANIE: {data_name} '{getattr(self, 'current_img_path_for_debug', 'N/A')}' mal tvar {original_shape}, po uprave na {target_shape} ma {img_numpy.shape}. Moze dojst k chybe.")
        return img_numpy

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        # index: index vzorky
        self.current_img_path_for_debug = self.image_list[index]
        img_path, lbl_path = self.image_list[index], self.image_list[index].replace('images','labels').replace('wrappedbg','unwrapped')

        if not os.path.exists(img_path):
            print(f"CHYBA: Vstupny obrazok neexistuje: {img_path}"); return None, None, None
        if not os.path.exists(lbl_path):
            print(f"CHYBA: Label obrazok neexistuje: {lbl_path}"); return None, None, None

        try:
            wrapped_orig_phase, unwrapped_orig = tiff.imread(img_path), tiff.imread(lbl_path)
        except Exception as e: print(f"CHYBA nacitania TIFF: {img_path} alebo {lbl_path}. Error: {e}"); return None,None,None

        wrapped_orig_phase = self._ensure_shape_and_type(wrapped_orig_phase, self.target_img_size, "Wrapped phase")
        unwrapped_orig = self._ensure_shape_and_type(unwrapped_orig, self.target_img_size, "Unwrapped phase")

        if self.input_processing_type == 'sincos':
            sin_phi = np.sin(wrapped_orig_phase)
            cos_phi = np.cos(wrapped_orig_phase)
            wrapped_input_numpy = np.stack([sin_phi, cos_phi], axis=0)
            wrapped_input_tensor = torch.from_numpy(wrapped_input_numpy.copy())
        elif self.input_processing_type == 'direct_minmax':
            wrapped_tensor_orig = torch.from_numpy(wrapped_orig_phase.copy())
            wrapped_norm_minmax = self._normalize_input_minmax_to_minus_one_one(wrapped_tensor_orig, self.input_min, self.input_max)
            wrapped_input_tensor = wrapped_norm_minmax.unsqueeze(0)
        else: raise ValueError(f"Neznamy input_processing_type: {self.input_processing_type}")

        unwrapped_tensor_orig = torch.from_numpy(unwrapped_orig.copy())
        unwrapped_norm_zscore = self._normalize_target_z_score(unwrapped_tensor_orig, self.target_mean, self.target_std)
        unwrapped_target_tensor = unwrapped_norm_zscore.unsqueeze(0)

        return wrapped_input_tensor, unwrapped_target_tensor, torch.from_numpy(unwrapped_orig.copy()).unsqueeze(0)

# --- POMOCNE FUNKCIE PRE DATA ---
def collate_fn_skip_none(batch):
    # batch: zoznam vzoriek z DataLoaderu
    batch = list(filter(lambda x: x is not None and x[0] is not None and x[1] is not None and x[2] is not None, batch))
    if not batch: return None, None, None
    return torch.utils.data.dataloader.default_collate(batch)

def denormalize_target_z_score(data_norm, original_mean, original_std):
    # data_norm: normalizovane data (torch.Tensor)
    # original_mean: povodny priemer pred normalizaciou
    # original_std: povodna standardna odchylka pred normalizaciou
    if original_std < 1e-6: return torch.full_like(data_norm, original_mean)
    return data_norm * original_std + original_mean

# --- METRIKY KVALITY OBRAZU ---
try:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    print("Kniznica scikit-image nie je nainstalovana. PSNR a SSIM nebudu vypocitane.")
    print("Nainstaluj ju pomocou: pip install scikit-image")
    SKIMAGE_AVAILABLE = False

def calculate_psnr_ssim(gt_img_numpy, pred_img_numpy):
    # gt_img_numpy: referencny obrazok (numpy array)
    # pred_img_numpy: predikovany obrazok (numpy array)
    if not SKIMAGE_AVAILABLE:
        return np.nan, np.nan

    gt_img_numpy = gt_img_numpy.squeeze()
    pred_img_numpy = pred_img_numpy.squeeze()

    data_range = gt_img_numpy.max() - gt_img_numpy.min()
    if data_range < 1e-6:
        current_psnr = float('inf') if np.allclose(gt_img_numpy, pred_img_numpy) else 0.0
    else:
        current_psnr = psnr(gt_img_numpy, pred_img_numpy, data_range=data_range)

    min_dim = min(gt_img_numpy.shape)
    win_size = min(7, min_dim)
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        current_ssim = np.nan
    else:
        try:
            current_ssim = ssim(gt_img_numpy, pred_img_numpy, data_range=data_range, channel_axis=None, win_size=win_size, gaussian_weights=True, use_sample_covariance=False)
        except ValueError:
            current_ssim = np.nan

    return current_psnr, current_ssim

def calculate_gdl_metric(pred_img_numpy, gt_img_numpy):
    # pred_img_numpy: predikovany obrazok (numpy array, 2D)
    # gt_img_numpy: referencny obrazok (numpy array, 2D)
    pred_img_numpy = pred_img_numpy.astype(np.float32)
    gt_img_numpy = gt_img_numpy.astype(np.float32)

    if pred_img_numpy.ndim != 2 or gt_img_numpy.ndim != 2:
        return np.nan

    pred_grad_x = pred_img_numpy[:, 1:] - pred_img_numpy[:, :-1]
    gt_grad_x = gt_img_numpy[:, 1:] - gt_img_numpy[:, :-1]

    pred_grad_y = pred_img_numpy[1:, :] - pred_img_numpy[:-1, :]
    gt_grad_y = gt_img_numpy[1:, :] - gt_img_numpy[:-1, :]

    if pred_grad_x.size == 0 or pred_grad_y.size == 0:
        return np.nan

    gdl_x = np.mean(np.abs(pred_grad_x - gt_grad_x))
    gdl_y = np.mean(np.abs(pred_grad_y - gt_grad_y))

    return (gdl_x + gdl_y) / 2.0

# --- HLAVNA EVALUACNA FUNKCIA ---
def evaluate_and_visualize_model(
    config_path,
    weights_path,
    test_dataset_path,
    device_str='cuda'
    ):
    # config_path: cesta ku konfiguracnemu suboru (.txt)
    # weights_path: cesta k suboru s vahami modelu (.pth)
    # test_dataset_path: cesta k adresaru s testovacimi datami
    # device_str: zariadenie na pouzitie ('cuda' alebo 'cpu')
    if not os.path.exists(config_path):
        print(f"CHYBA: Konfiguracny subor nebol najdeny: {config_path}")
        return
    if not os.path.exists(weights_path):
        print(f"CHYBA: Subor s vahami nebol najdeny: {weights_path}")
        return

    config = {}
    with open(config_path, 'r') as f:
        for line in f:
            if ":" in line:
                key, value = line.split(":", 1)
                config[key.strip()] = value.strip()

    print("--- Nacitana Konfiguracia Experimentu ---")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("-" * 30)

    encoder_name = config.get("Encoder Name", "resnet18")
    input_processing_type = config.get("Input Processing", "sincos")
    target_original_mean = float(config.get("Target Norm (Z-score) Mean", 0.0))
    target_original_std = float(config.get("Target Norm (Z-score) Std", 1.0))
    if target_original_std < 1e-6:
        print("VAROVANIE: Nacitana Std pre ciel je velmi nizka.")

    input_original_min_max = None
    if input_processing_type == 'direct_minmax':
        input_min = float(config.get("Input Norm (MinMax) Min", -np.pi))
        input_max = float(config.get("Input Norm (MinMax) Max", np.pi))
        if input_min == input_max:
            print("VAROVANIE: Nacitane Min a Max pre vstup su rovnake.")
        input_original_min_max = (input_min, input_max)

    print(f"\nNacitavam testovaci dataset z: {test_dataset_path}")
    test_dataset = CustomDataset(
        path_to_data=test_dataset_path,
        input_processing_type=input_processing_type,
        norm_stats_input=input_original_min_max,
        norm_stats_target=(target_original_mean, target_original_std),
        augmentation_strength='none',
        is_train_set=False
    )
    if len(test_dataset) == 0:
        print("CHYBA: Testovaci dataset je prazdny.")
        return

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn_skip_none)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Pouzivam zariadenie: {device}")

    in_channels_for_model = 1 if input_processing_type == 'direct_minmax' else 2

    model_architecture_name_from_config = config.get("Model Architecture")
    if model_architecture_name_from_config and model_architecture_name_from_config.lower() != "smp_unet":
        print(f"VAROVANIE: Konfiguracny subor specifikuje architekturu '{model_architecture_name_from_config}', "
              f"ale tento skript je nastaveny na pouzitie 'smp.Unet' s enkoderom '{encoder_name}'.")

    try:
        net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels_for_model,
            classes=1,
            activation=None
        ).to(device)
        print(f"Pouzivam smp.Unet s enkoderom: {encoder_name}")
    except Exception as e:
        print(f"CHYBA pri inicializacii smp.Unet s enkoderom '{encoder_name}': {e}")
        return

    try:
        net.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"Vahy modelu uspesne nacitane z: {weights_path}")
    except Exception as e:
        print(f"CHYBA pri nacitani vah modelu: {e}")
        return
    net.eval()

    print("\nEvaluujem cely testovaci dataset...")
    all_mae_denorm = []
    all_mse_denorm = []
    all_psnr = []
    all_ssim = []

    best_mae_sample_info = {"mae": float('inf'), "index": -1, "pred_denorm": None, "gt_denorm": None, "input_orig": None}
    worst_mae_sample_info = {"mae": -1.0, "index": -1, "pred_denorm": None, "gt_denorm": None, "input_orig": None}
    avg_mae_sample_info = {"mae": -1.0, "index": -1, "pred_denorm": None, "gt_denorm": None, "input_orig": None, "diff_from_avg_mae": float('inf')}

    all_pixel_errors_flat = []
    all_samples_data_for_avg = []

    with torch.no_grad():
        for i, batch_data in tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluacia batchov"):
            if batch_data is None or batch_data[0] is None:
                print(f"Preskakujem chybny batch v teste, iteracia {i}")
                continue

            input_norm_batch, target_norm_batch, target_orig_batch = batch_data
            input_norm_batch = input_norm_batch.to(device)

            pred_norm_batch = net(input_norm_batch)

            for k in range(pred_norm_batch.size(0)):
                current_sample_idx = i * test_loader.batch_size + k
                if current_sample_idx >= len(test_dataset): continue

                pred_norm_sample_tensor = pred_norm_batch[k]
                target_orig_sample_numpy = target_orig_batch[k].cpu().numpy().squeeze()

                pred_denorm_sample_tensor = denormalize_target_z_score(pred_norm_sample_tensor, target_original_mean, target_original_std)
                pred_denorm_sample_numpy = pred_denorm_sample_tensor.cpu().numpy().squeeze()

                current_pixel_errors = np.abs(pred_denorm_sample_numpy - target_orig_sample_numpy)
                all_pixel_errors_flat.extend(current_pixel_errors.flatten().tolist())

                mae = np.mean(current_pixel_errors)
                mse = np.mean((pred_denorm_sample_numpy - target_orig_sample_numpy)**2)

                all_mae_denorm.append(mae)
                all_mse_denorm.append(mse)

                img_path_current = test_dataset.image_list[current_sample_idx]
                current_input_orig = tiff.imread(img_path_current)
                all_samples_data_for_avg.append((mae, current_input_orig, target_orig_sample_numpy, pred_denorm_sample_numpy, current_sample_idx))

                if SKIMAGE_AVAILABLE:
                    psnr_val, ssim_val = calculate_psnr_ssim(target_orig_sample_numpy, pred_denorm_sample_numpy)
                    if not np.isnan(psnr_val): all_psnr.append(psnr_val)
                    if not np.isnan(ssim_val): all_ssim.append(ssim_val)

                if mae < best_mae_sample_info["mae"]:
                    best_mae_sample_info["mae"] = mae
                    best_mae_sample_info["index"] = current_sample_idx
                    best_mae_sample_info["input_orig"] = current_input_orig
                    best_mae_sample_info["gt_denorm"] = target_orig_sample_numpy
                    best_mae_sample_info["pred_denorm"] = pred_denorm_sample_numpy

                if mae > worst_mae_sample_info["mae"]:
                    worst_mae_sample_info["mae"] = mae
                    worst_mae_sample_info["index"] = current_sample_idx
                    worst_mae_sample_info["input_orig"] = current_input_orig
                    worst_mae_sample_info["gt_denorm"] = target_orig_sample_numpy
                    worst_mae_sample_info["pred_denorm"] = pred_denorm_sample_numpy

    avg_mae = np.mean(all_mae_denorm) if all_mae_denorm else np.nan
    avg_mse = np.mean(all_mse_denorm) if all_mse_denorm else np.nan
    avg_psnr = np.mean(all_psnr) if all_psnr else np.nan
    avg_ssim = np.mean(all_ssim) if all_ssim else np.nan

    print("\n--- Celkove Priemerne Metriky na Testovacom Datasete ---")
    print(f"Priemerna MAE (denormalizovana): {avg_mae:.4f}")
    print(f"Priemerna MSE (denormalizovana): {avg_mse:.4f}")
    if SKIMAGE_AVAILABLE:
        print(f"Priemerny PSNR: {avg_psnr:.2f} dB")
        print(f"Priemerny SSIM: {avg_ssim:.4f}")

    if not np.isnan(avg_mae) and all_samples_data_for_avg:
        min_diff_to_avg_mae = float('inf')
        avg_candidate_data = None
        for sample_mae_val, s_input_orig, s_gt_denorm, s_pred_denorm, s_idx in all_samples_data_for_avg:
            diff = abs(sample_mae_val - avg_mae)
            if diff < min_diff_to_avg_mae:
                min_diff_to_avg_mae = diff
                avg_candidate_data = (sample_mae_val, s_input_orig, s_gt_denorm, s_pred_denorm, s_idx)

        if avg_candidate_data:
            avg_mae_sample_info["mae"] = avg_candidate_data[0]
            avg_mae_sample_info["input_orig"] = avg_candidate_data[1]
            avg_mae_sample_info["gt_denorm"] = avg_candidate_data[2]
            avg_mae_sample_info["pred_denorm"] = avg_candidate_data[3]
            avg_mae_sample_info["index"] = avg_candidate_data[4]
            avg_mae_sample_info["diff_from_avg_mae"] = min_diff_to_avg_mae

    print("\n--- Extremne Hodnoty MAE (len logovanie) ---")
    if best_mae_sample_info["index"] != -1:
        print(f"Najlepsia MAE na vzorke: {best_mae_sample_info['mae']:.4f} (index: {best_mae_sample_info['index']})")
    if avg_mae_sample_info["index"] != -1:
        print(f"Vzorka najblizsie k priemernej MAE ({avg_mae:.4f}): MAE={avg_mae_sample_info['mae']:.4f} (index: {avg_mae_sample_info['index']}, rozdiel: {avg_mae_sample_info['diff_from_avg_mae']:.4f})")
    if worst_mae_sample_info["index"] != -1:
        print(f"Najhorsia MAE na vzorke: {worst_mae_sample_info['mae']:.4f} (index: {worst_mae_sample_info['index']})")

    run_name_for_files = os.path.splitext(os.path.basename(weights_path))[0].replace("best_weights_", "")

    if best_mae_sample_info["index"] != -1 or worst_mae_sample_info["index"] != -1 or avg_mae_sample_info["index"] != -1:
        extreme_mae_log_path = f"extreme_mae_values_{run_name_for_files}.txt"
        with open(extreme_mae_log_path, 'w') as f:
            f.write(f"Experiment: {run_name_for_files}\n")
            f.write("--- Extremne a Priemerne Hodnoty MAE ---\n")
            if best_mae_sample_info["index"] != -1:
                f.write(f"Najlepsia MAE: {best_mae_sample_info['mae']:.6f} (index: {best_mae_sample_info['index']})\n")
            else:
                f.write("Najlepsia MAE: N/A\n")

            if avg_mae_sample_info["index"] != -1:
                f.write(f"Vzorka najblizsie k priemernej MAE ({avg_mae:.6f}): MAE={avg_mae_sample_info['mae']:.6f} (index: {avg_mae_sample_info['index']}, rozdiel od priemeru: {avg_mae_sample_info['diff_from_avg_mae']:.6f})\n")
            else:
                f.write("Vzorka najblizsie k priemernej MAE: N/A\n")

            if worst_mae_sample_info["index"] != -1:
                f.write(f"Najhorsia MAE: {worst_mae_sample_info['mae']:.6f} (index: {worst_mae_sample_info['index']})\n")
            else:
                f.write("Najhorsia MAE: N/A\n")
            f.write(f"\nPriemerna MAE (cely dataset): {avg_mae:.6f}\n")
        print(f"Extremne a priemerne MAE hodnoty ulozene do: {extreme_mae_log_path}")

    if all_pixel_errors_flat:
        all_pixel_errors_flat_np = np.array(all_pixel_errors_flat)
        plt.figure(figsize=(12, 7))
        plt.hist(all_pixel_errors_flat_np, bins=100, color='dodgerblue', edgecolor='black', alpha=0.7)
        plt.title('Histogram Absolutnych Chyb Predikcie (vsetky pixely)', fontsize=16)
        plt.xlabel('Absolutna Chyba (radiany)', fontsize=14)
        plt.ylabel('Pocet Pixelov', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.yscale('log')
        hist_save_path_png = f"error_histogram_{run_name_for_files}.png"
        hist_save_path_svg = f"error_histogram_{run_name_for_files}.svg"
        plt.savefig(hist_save_path_png)
        plt.savefig(hist_save_path_svg)
        print(f"Histogram chyb ulozeny do: {hist_save_path_png} a {hist_save_path_svg}")
        plt.show()
        plt.close()

    if best_mae_sample_info["index"] != -1 and worst_mae_sample_info["index"] != -1 and avg_mae_sample_info["index"] != -1:
        print(f"\nVizualizujem a ukladam najlepsi, priemerny a najhorsi MAE pripad (nove usporiadanie)...")

        samples_to_plot = [
            ("Min MAE", best_mae_sample_info),
            ("Avg MAE", avg_mae_sample_info),
            ("Max MAE", worst_mae_sample_info)
        ]

        all_input_orig_col1 = []
        all_gt_pred_col23 = []

        for _, sample_info in samples_to_plot:
            all_input_orig_col1.append(sample_info["input_orig"])
            all_gt_pred_col23.append(sample_info["gt_denorm"])
            all_gt_pred_col23.append(sample_info["pred_denorm"])

        vmin_col1 = np.min([img.min() for img in all_input_orig_col1]) if all_input_orig_col1 else 0
        vmax_col1 = np.max([img.max() for img in all_input_orig_col1]) if all_input_orig_col1 else 1

        vmin_col23 = np.min([img.min() for img in all_gt_pred_col23]) if all_gt_pred_col23 else 0
        vmax_col23 = np.max([img.max() for img in all_gt_pred_col23]) if all_gt_pred_col23 else 1

        fig, axs = plt.subplots(3, 4, figsize=(16, 13))

        row_titles = ["Min MAE", "Avg MAE", "Max MAE"]
        col_titles = ["Zabaleny obraz", "Rozbaleny referencny obraz", "Predikcia", "Absolutna chyba"]

        error_map_mappables = []

        for i, (row_title_text, sample_info) in enumerate(samples_to_plot):
            img0 = axs[i, 0].imshow(sample_info["input_orig"], cmap='gray', vmin=vmin_col1, vmax=vmax_col1)
            img1 = axs[i, 1].imshow(sample_info["gt_denorm"], cmap='gray', vmin=vmin_col23, vmax=vmax_col23)
            img2 = axs[i, 2].imshow(sample_info["pred_denorm"], cmap='gray', vmin=vmin_col23, vmax=vmax_col23)

            error_map = np.abs(sample_info["pred_denorm"] - sample_info["gt_denorm"])
            img3 = axs[i, 3].imshow(error_map, cmap='viridis')
            error_map_mappables.append(img3)

        for j, col_title_text in enumerate(col_titles):
            axs[0, j].set_title(col_title_text, fontsize=16, pad=20)

        for ax_row in axs:
            for ax in ax_row:
                ax.axis('off')

        for i in range(3):
            fig.colorbar(error_map_mappables[i], ax=axs[i, 3], orientation='vertical', fraction=0.046, pad=0.02, aspect=15)

        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.18, top=0.90, wspace=0.0, hspace=0.06)

        pos_col0_ax2 = axs[2,0].get_position()
        cax1_left = pos_col0_ax2.x0
        cax1_bottom = 0.13
        cax1_width = pos_col0_ax2.width
        cax1_height = 0.025
        cax1 = fig.add_axes([cax1_left, cax1_bottom, cax1_width, cax1_height])
        cb1 = fig.colorbar(img0, cax=cax1, orientation='horizontal')

        pos_col1_ax2 = axs[2,1].get_position()
        pos_col2_ax2 = axs[2,2].get_position()

        cax23_left = pos_col1_ax2.x0
        cax23_bottom = 0.13
        cax23_width = (pos_col2_ax2.x0 + pos_col2_ax2.width) - pos_col1_ax2.x0
        cax23_height = 0.025
        cax23 = fig.add_axes([cax23_left, cax23_bottom, cax23_width, cax23_height])
        cb23 = fig.colorbar(img1, cax=cax23, orientation='horizontal')

        base_save_name = f"detailed_comparison_mae_errors_{run_name_for_files}"
        save_fig_path_png = f"{base_save_name}.png"
        save_fig_path_svg = f"{base_save_name}.svg"

        plt.savefig(save_fig_path_png, dpi=200)
        print(f"Detailna vizualizacia (nove usporiadanie) ulozena do: {save_fig_path_png}")

        plt.savefig(save_fig_path_svg)
        print(f"Detailna vizualizacia (nove usporiadanie) ulozena aj do: {save_fig_path_svg}")

        plt.show()
        plt.close()
    else:
        print("Nepodarilo sa najst dostatok dat pre plnu detailnu vizualizaciu.")

# --- SPUSTENIE SKRIPTU ---
if __name__ == "__main__":
    config_file_path = r"C:\\Users\\juraj\\Desktop\\TRENOVANIE_bakalarka_simul\\optimalizacia_hype\\trenovanie_5\\config_R34imgnet_direct_MAE_GDL0p3_AugMedium_LR5em04_WD1em04_Ep120_ESp30_Tmax120_EtaMin1em07_bs8_bs8.txt"
    weights_file_path = r"C:\\Users\\juraj\\Desktop\\TRENOVANIE_bakalarka_simul\\optimalizacia_hype\\trenovanie_5\\best_weights_R34imgnet_direct_MAE_GDL0p3_AugMedium_LR5em04_WD1em04_Ep120_ESp30_Tmax120_EtaMin1em07_bs8_bs8.pth"
    test_dataset_directory = r"C:\\Users\\juraj\\Desktop\\TRENOVANIE_bakalarka_simul\\split_dataset_tiff_for_dynamic_v_stratified_final\\static_test_dataset"

    if not os.path.exists(config_file_path):
        print(f"CHYBA: Konfiguracny subor neexistuje: {config_file_path}")
    elif not os.path.exists(weights_file_path):
        print(f"CHYBA: Subor s vahami neexistuje: {weights_file_path}")
    elif not os.path.isdir(test_dataset_directory):
        print(f"CHYBA: Adresar testovacieho datasetu neexistuje: {test_dataset_directory}")
    else:
        evaluate_and_visualize_model(
            config_path=config_file_path,
            weights_path=weights_file_path,
            test_dataset_path=test_dataset_directory,
            device_str='cuda'
        )
        
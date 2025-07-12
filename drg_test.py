import os
import glob
import time  # Aj keď tu nebude priamo meraný čas tréningu, môže sa hodiť
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim  # Pre testovanie nie je potrebný optimizer
import torch.nn.functional as F  # Môže byť potrebný, ak ho používaš v modeli alebo loss
import tifffile as tiff
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as T  # Ak používaš augmentácie v datasete (pre test nie)

# Ak používaš smp.Unet, importuj ho. Ak vlastnú architektúru, definuj ju.
import segmentation_models_pytorch as smp
from tqdm import tqdm

# Importuj aj vlastnú architektúru, ak ju používaš
# from tvoj_modul_s_architekturou import CustomResUNet

# --- Globálne Konštanty (ak sú potrebné pre CustomDataset alebo augmentácie) ---
NORMALIZED_INPUT_CLAMP_MIN = -1.0
NORMALIZED_INPUT_CLAMP_MAX = 1.0
NORMALIZED_INPUT_ERASING_VALUE = 0.0

# ----------------------------------------------------------------------------------
# KOPÍRUJ SEM DEFINÍCIE SVOJICH TRIED A FUNKCIÍ Z TRÉNINGOVÉHO SKRIPTU:
# - AddGaussianNoiseTransform (ak bola súčasťou `CustomDataset` a potrebuješ ju na rekonštrukciu defaultných aug. nastavení)
# - CustomDataset (presne tá verzia, s ktorou bol model trénovaný!)
# - collate_fn_skip_none
# - denormalize_target_z_score
# - Ak používaš vlastnú architektúru (napr. CustomResUNet), skopíruj sem aj jej definíciu.
# ----------------------------------------------------------------------------------


# Príklad štruktúry CustomDataset (musíš si ju doplniť svojou plnou verziou)
class CustomDataset(Dataset):
    def __init__(
        self,
        path_to_data,
        input_processing_type,
        norm_stats_input,
        norm_stats_target,
        augmentation_strength="none",
        is_train_set=False,
        target_img_size=(512, 512),
    ):
        self.path = path_to_data
        self.image_list = sorted(glob.glob(os.path.join(self.path, "images", "*.tiff")))
        self.input_processing_type = input_processing_type
        self.input_min, self.input_max = (None, None)
        if input_processing_type == "direct_minmax" and norm_stats_input:
            self.input_min, self.input_max = norm_stats_input
        elif input_processing_type == "sincos" and norm_stats_input:
            print(
                "Poznámka: norm_stats_input poskytnuté pre sincos, ale zvyčajne sa nepoužívajú na normalizáciu sin/cos."
            )

        self.target_mean, self.target_std = norm_stats_target
        self.is_train_set = is_train_set
        self.target_img_size = target_img_size
        self.augmentation_strength = augmentation_strength

        self.geometric_transforms = (
            None  # Pre testovanie sa augmentácie zvyčajne nepoužívajú
        )
        self.pixel_transforms = (
            None  # Pre testovanie sa augmentácie zvyčajne nepoužívajú
        )

        # _setup_augmentations sa pre test_dataset s augmentation_strength='none' typicky nevolá
        # alebo by malo byť prázdne. Ak by ste chceli mať plnú definíciu pre konzistenciu:
        # if self.is_train_set and self.augmentation_strength != 'none':
        #     self._setup_augmentations(self.augmentation_strength)

    def _setup_augmentations(self, strength):
        pass  # Pre test nepotrebujeme nastavovať augmentácie

    def _normalize_input_minmax_to_minus_one_one(self, data, min_val, max_val):
        if max_val == min_val:
            return torch.zeros_like(data)
        return 2.0 * (data - min_val) / (max_val - min_val) - 1.0

    def _normalize_target_z_score(self, data, mean_val, std_val):
        if std_val < 1e-6:
            return data - mean_val
        return (data - mean_val) / std_val

    def _ensure_shape_and_type(
        self, img_numpy, target_shape, data_name="Image", dtype=np.float32
    ):
        img_numpy = img_numpy.astype(dtype)
        if img_numpy.shape[-2:] != target_shape:  # Kontroluje len H, W dimenzie
            original_shape = img_numpy.shape
            # Ak je menší, padneme
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
                    img_numpy = np.pad(
                        img_numpy,
                        ((pad_top, pad_bottom), (pad_left, pad_right)),
                        mode="reflect",
                    )
                elif img_numpy.ndim == 3:  # Predpoklad (C, H, W)
                    img_numpy = np.pad(
                        img_numpy,
                        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                        mode="reflect",
                    )

            # Ak je väčší, orežeme (center crop)
            h, w = img_numpy.shape[-2:]  # Znovu získame rozmery po prípadnom paddingu
            if h > target_h or w > target_w:
                start_h = (h - target_h) // 2
                start_w = (w - target_w) // 2
                if img_numpy.ndim == 2:
                    img_numpy = img_numpy[
                        start_h : start_h + target_h, start_w : start_w + target_w
                    ]
                elif img_numpy.ndim == 3:
                    img_numpy = img_numpy[
                        :, start_h : start_h + target_h, start_w : start_w + target_w
                    ]

            if img_numpy.shape[-2:] != target_shape:  # Finálna kontrola
                print(
                    f"VAROVANIE: {data_name} '{getattr(self, 'current_img_path_for_debug', 'N/A')}' mal tvar {original_shape}, po úprave na {target_shape} má {img_numpy.shape}. Môže dôjsť k chybe."
                )
                # V prípade pretrvávajúcej nezhody by sa mala vyhodiť chyba, alebo použiť robustnejší resize.
                # Pre teraz ponechávam, ale je to miesto na pozornosť.
                # raise ValueError(f"{data_name} '{getattr(self, 'current_img_path_for_debug', 'N/A')}' má tvar {img_numpy.shape} po úprave, očakáva sa H,W ako {target_shape}")
        return img_numpy

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        self.current_img_path_for_debug = self.image_list[index]
        img_path, lbl_path = self.image_list[index], self.image_list[index].replace(
            "images", "labels"
        ).replace("wrappedbg", "unwrapped")

        # Kontrola existencie súborov
        if not os.path.exists(img_path):
            print(f"CHYBA: Vstupný obrázok neexistuje: {img_path}")
            return None, None, None
        if not os.path.exists(lbl_path):
            print(f"CHYBA: Label obrázok neexistuje: {lbl_path}")
            return None, None, None

        try:
            wrapped_orig_phase, unwrapped_orig = tiff.imread(img_path), tiff.imread(
                lbl_path
            )
        except Exception as e:
            print(f"CHYBA načítania TIFF: {img_path} alebo {lbl_path}. Error: {e}")
            return None, None, None

        wrapped_orig_phase = self._ensure_shape_and_type(
            wrapped_orig_phase, self.target_img_size, "Wrapped phase"
        )
        unwrapped_orig = self._ensure_shape_and_type(
            unwrapped_orig, self.target_img_size, "Unwrapped phase"
        )

        if self.input_processing_type == "sincos":
            sin_phi = np.sin(wrapped_orig_phase)
            cos_phi = np.cos(wrapped_orig_phase)
            wrapped_input_numpy = np.stack(
                [sin_phi, cos_phi], axis=0
            )  # Shape (2, H, W)
            wrapped_input_tensor = torch.from_numpy(wrapped_input_numpy.copy())
        elif self.input_processing_type == "direct_minmax":
            wrapped_tensor_orig = torch.from_numpy(
                wrapped_orig_phase.copy()
            )  # Shape (H, W)
            wrapped_norm_minmax = self._normalize_input_minmax_to_minus_one_one(
                wrapped_tensor_orig, self.input_min, self.input_max
            )
            wrapped_input_tensor = wrapped_norm_minmax.unsqueeze(0)  # Shape (1, H, W)
        else:
            raise ValueError(
                f"Neznámy input_processing_type: {self.input_processing_type}"
            )

        unwrapped_tensor_orig = torch.from_numpy(unwrapped_orig.copy())  # Shape (H, W)
        unwrapped_norm_zscore = self._normalize_target_z_score(
            unwrapped_tensor_orig, self.target_mean, self.target_std
        )
        unwrapped_target_tensor = unwrapped_norm_zscore.unsqueeze(0)  # Shape (1, H, W)

        return (
            wrapped_input_tensor,
            unwrapped_target_tensor,
            torch.from_numpy(unwrapped_orig.copy()).unsqueeze(0),
        )  # Vraciame aj pôvodný GT (1,H,W)


def collate_fn_skip_none(batch):
    batch = list(
        filter(
            lambda x: x is not None
            and x[0] is not None
            and x[1] is not None
            and x[2] is not None,
            batch,
        )
    )
    if not batch:
        return None, None, None
    return torch.utils.data.dataloader.default_collate(batch)


# Denormalizačná funkcia
def denormalize_target_z_score(data_norm, original_mean, original_std):
    if original_std < 1e-6:
        return torch.full_like(data_norm, original_mean)
    return data_norm * original_std + original_mean


# Metriky PSNR a SSIM (vyžaduje skimage)
try:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

    SKIMAGE_AVAILABLE = True
except ImportError:
    print("Knižnica scikit-image nie je nainštalovaná. PSNR a SSIM nebudú vypočítané.")
    print("Nainštaluj ju pomocou: pip install scikit-image")
    SKIMAGE_AVAILABLE = False


def calculate_psnr_ssim(gt_img_numpy, pred_img_numpy):
    if not SKIMAGE_AVAILABLE:
        return np.nan, np.nan

    gt_img_numpy = gt_img_numpy.squeeze()
    pred_img_numpy = pred_img_numpy.squeeze()

    data_range = gt_img_numpy.max() - gt_img_numpy.min()
    if data_range < 1e-6:
        current_psnr = (
            float("inf") if np.allclose(gt_img_numpy, pred_img_numpy) else 0.0
        )
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
            current_ssim = ssim(
                gt_img_numpy,
                pred_img_numpy,
                data_range=data_range,
                channel_axis=None,
                win_size=win_size,
                gaussian_weights=True,
                use_sample_covariance=False,
            )
        except ValueError:
            current_ssim = np.nan

    return current_psnr, current_ssim


# Nová funkcia pre výpočet GDL metriky
def calculate_gdl_metric(pred_img_numpy, gt_img_numpy):
    """Vypočíta Gradient Difference Loss (GDL) medzi dvoma obrázkami."""
    pred_img_numpy = pred_img_numpy.astype(np.float32)
    gt_img_numpy = gt_img_numpy.astype(np.float32)

    if pred_img_numpy.ndim != 2 or gt_img_numpy.ndim != 2:
        # print("Varovanie: GDL očakáva 2D obrázky.") # Môže byť príliš časté
        return np.nan  # Alebo vrátiť nejakú defaultnú vysokú hodnotu

    # Gradienty X
    pred_grad_x = pred_img_numpy[:, 1:] - pred_img_numpy[:, :-1]
    gt_grad_x = gt_img_numpy[:, 1:] - gt_img_numpy[:, :-1]

    # Gradienty Y
    pred_grad_y = pred_img_numpy[1:, :] - pred_img_numpy[:-1, :]
    gt_grad_y = gt_img_numpy[1:, :] - gt_img_numpy[:-1, :]

    if (
        pred_grad_x.size == 0 or pred_grad_y.size == 0
    ):  # Ak je obrázok príliš malý (1px široký/vysoký)
        return np.nan

    gdl_x = np.mean(np.abs(pred_grad_x - gt_grad_x))
    gdl_y = np.mean(np.abs(pred_grad_y - gt_grad_y))

    return (gdl_x + gdl_y) / 2.0


def evaluate_and_visualize_model(
    config_path,
    weights_path,
    test_dataset_path,
    # image_index_to_show, # Tento parameter už nebude priamo použitý pre hlavný graf
    device_str="cuda",
):
    """
    Načíta model, evaluuje ho na testovacom datasete a vizualizuje najlepší/najhorší MAE.
    """
    if not os.path.exists(config_path):
        print(f"CHYBA: Konfiguračný súbor nebol nájdený: {config_path}")
        return
    if not os.path.exists(weights_path):
        print(f"CHYBA: Súbor s váhami nebol nájdený: {weights_path}")
        return

    # --- Načítanie Konfigurácie ---
    config = {}
    with open(config_path, "r") as f:
        for line in tqdm(f):
            if ":" in line:
                key, value = line.split(":", 1)
                config[key.strip()] = value.strip()

    print("--- Načítaná Konfigurácia Experimentu ---")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("-" * 30)

    encoder_name = config.get("Encoder Name", "resnet18")
    input_processing_type = config.get("Input Processing", "sincos")
    target_original_mean = float(config.get("Target Norm (Z-score) Mean", 0.0))
    target_original_std = float(config.get("Target Norm (Z-score) Std", 1.0))
    if target_original_std < 1e-6:
        print("VAROVANIE: Načítaná Std pre cieľ je veľmi nízka.")

    input_original_min_max = None
    if input_processing_type == "direct_minmax":
        input_min = float(config.get("Input Norm (MinMax) Min", -np.pi))
        input_max = float(config.get("Input Norm (MinMax) Max", np.pi))
        if input_min == input_max:
            print("VAROVANIE: Načítané Min a Max pre vstup sú rovnaké.")
        input_original_min_max = (input_min, input_max)

    # --- Príprava Datasetu a DataLoaderu ---
    print(f"\nNačítavam testovací dataset z: {test_dataset_path}")
    test_dataset = CustomDataset(
        path_to_data=test_dataset_path,
        input_processing_type=input_processing_type,
        norm_stats_input=input_original_min_max,
        norm_stats_target=(target_original_mean, target_original_std),
        augmentation_strength="none",
        is_train_set=False,
    )
    if len(test_dataset) == 0:
        print("CHYBA: Testovací dataset je prázdny.")
        return

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_skip_none,
    )

    # --- Načítanie Modelu ---
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Používam zariadenie: {device}")

    in_channels_for_model = 1 if input_processing_type == "direct_minmax" else 2

    model_architecture_name_from_config = config.get("Model Architecture")
    if (
        model_architecture_name_from_config
        and model_architecture_name_from_config.lower() != "smp_unet"
    ):
        print(
            f"VAROVANIE: Konfiguračný súbor špecifikuje architektúru '{model_architecture_name_from_config}', "
            f"ale tento skript je nastavený na použitie 'smp.Unet' s enkodérom '{encoder_name}'."
        )

    try:
        net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels_for_model,
            classes=1,
            activation=None,
        ).to(device)
        print(f"Používam smp.Unet s enkóderom: {encoder_name}")
    except Exception as e:
        print(f"CHYBA pri inicializácii smp.Unet s enkodérom '{encoder_name}': {e}")
        return

    try:
        net.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=True)
        )
        print(f"Váhy modelu úspešne načítané z: {weights_path}")
    except Exception as e:
        print(f"CHYBA pri načítaní váh modelu: {e}")
        return
    net.eval()

    # --- Evaluácia Celého Testovacieho Setu ---
    print("\nEvaluujem celý testovací dataset...")
    all_mae_denorm = []
    all_mse_denorm = []
    all_psnr = []
    all_ssim = []

    # Sledovanie najlepšej a najhoršej MAE
    # Inicializujeme indexy na -1, aby sme vedeli, či sme našli nejaké vzorky
    best_mae_sample_info = {
        "mae": float("inf"),
        "index": -1,
        "pred_denorm": None,
        "gt_denorm": None,
        "input_orig": None,
    }
    worst_mae_sample_info = {
        "mae": -1.0,
        "index": -1,
        "pred_denorm": None,
        "gt_denorm": None,
        "input_orig": None,
    }

    with torch.no_grad():
        for i, batch_data in tqdm(enumerate(test_loader)):
            if batch_data is None or batch_data[0] is None:
                print(f"Preskakujem chybný batch v teste, iterácia {i}")
                continue

            input_norm_batch, target_norm_batch, target_orig_batch = batch_data
            input_norm_batch = input_norm_batch.to(device)

            pred_norm_batch = net(input_norm_batch)

            for k in tqdm(range(pred_norm_batch.size(0))):
                current_sample_idx = i * test_loader.batch_size + k
                if current_sample_idx >= len(test_dataset):
                    continue

                pred_norm_sample_tensor = pred_norm_batch[k]
                target_orig_sample_numpy = target_orig_batch[k].cpu().numpy().squeeze()

                pred_denorm_sample_tensor = denormalize_target_z_score(
                    pred_norm_sample_tensor, target_original_mean, target_original_std
                )
                pred_denorm_sample_numpy = (
                    pred_denorm_sample_tensor.cpu().numpy().squeeze()
                )

                mae = np.mean(
                    np.abs(pred_denorm_sample_numpy - target_orig_sample_numpy)
                )
                mse = np.mean(
                    (pred_denorm_sample_numpy - target_orig_sample_numpy) ** 2
                )

                all_mae_denorm.append(mae)
                all_mse_denorm.append(mse)

                if SKIMAGE_AVAILABLE:
                    psnr_val, ssim_val = calculate_psnr_ssim(
                        target_orig_sample_numpy, pred_denorm_sample_numpy
                    )
                    if not np.isnan(psnr_val):
                        all_psnr.append(psnr_val)
                    if not np.isnan(ssim_val):
                        all_ssim.append(ssim_val)

                # Aktualizácia najlepšej MAE
                if mae < best_mae_sample_info["mae"]:
                    best_mae_sample_info["mae"] = mae
                    best_mae_sample_info["index"] = current_sample_idx
                    # Uložíme si potrebné dáta pre neskoršie vykreslenie
                    # Načítame pôvodný zabalený vstup pre túto vzorku
                    img_path_best = test_dataset.image_list[current_sample_idx]
                    best_mae_sample_info["input_orig"] = tiff.imread(img_path_best)
                    best_mae_sample_info["gt_denorm"] = target_orig_sample_numpy
                    best_mae_sample_info["pred_denorm"] = pred_denorm_sample_numpy

                # Aktualizácia najhoršej MAE
                if mae > worst_mae_sample_info["mae"]:
                    worst_mae_sample_info["mae"] = mae
                    worst_mae_sample_info["index"] = current_sample_idx
                    # Uložíme si potrebné dáta pre neskoršie vykreslenie
                    img_path_worst = test_dataset.image_list[current_sample_idx]
                    worst_mae_sample_info["input_orig"] = tiff.imread(img_path_worst)
                    worst_mae_sample_info["gt_denorm"] = target_orig_sample_numpy
                    worst_mae_sample_info["pred_denorm"] = pred_denorm_sample_numpy

    avg_mae = np.mean(all_mae_denorm) if all_mae_denorm else np.nan
    avg_mse = np.mean(all_mse_denorm) if all_mse_denorm else np.nan
    avg_psnr = np.mean(all_psnr) if all_psnr else np.nan
    avg_ssim = np.mean(all_ssim) if all_ssim else np.nan

    print("\n--- Celkové Priemerné Metriky na Testovacom Datasete ---")
    print(f"Priemerná MAE (denormalizovaná): {avg_mae:.4f}")
    print(f"Priemerná MSE (denormalizovaná): {avg_mse:.4f}")
    if SKIMAGE_AVAILABLE:
        print(f"Priemerný PSNR: {avg_psnr:.2f} dB")
        print(f"Priemerný SSIM: {avg_ssim:.4f}")

    print("\n--- Extrémne Hodnoty MAE (len logovanie) ---")
    if best_mae_sample_info["index"] != -1:
        print(
            f"Najlepšia MAE na vzorke: {best_mae_sample_info['mae']:.4f} (index: {best_mae_sample_info['index']})"
        )
    if worst_mae_sample_info["index"] != -1:
        print(
            f"Najhoršia MAE na vzorke: {worst_mae_sample_info['mae']:.4f} (index: {worst_mae_sample_info['index']})"
        )

    # --- Vizualizácia Najlepšej a Najhoršej MAE ---
    if best_mae_sample_info["index"] != -1 and worst_mae_sample_info["index"] != -1:
        print(f"\nVizualizujem a ukladám najlepší a najhorší MAE prípad...")

        fig, axs = plt.subplots(2, 3, figsize=(18, 10))  # 2 riadky, 3 stĺpce
        # run_name_for_title = os.path.splitext(os.path.basename(weights_path))[0].replace("best_weights_", "")
        # fig.suptitle(f"Najlepšia a Najhoršia MAE Predikcia\nRun: {run_name_for_title}", fontsize=16, y=0.98) # ODSTRÁNENÝ ALEBO ZAKOMENTOVANÝ RIADOK

        # Riadok 1: Najlepšia MAE
        axs[0, 0].imshow(best_mae_sample_info["input_orig"], cmap="gray")
        axs[0, 0].set_title("Zabalený obraz", fontsize=14)
        axs[0, 0].axis("off")

        im_gt_best = axs[0, 1].imshow(best_mae_sample_info["gt_denorm"], cmap="gray")
        axs[0, 1].set_title("Rozbalený referenčný obraz", fontsize=14)
        axs[0, 1].axis("off")
        # fig.colorbar(im_gt_best, ax=axs[0, 1], fraction=0.046, pad=0.04)

        common_min_best = min(
            best_mae_sample_info["gt_denorm"].min(),
            best_mae_sample_info["pred_denorm"].min(),
        )
        common_max_best = max(
            best_mae_sample_info["gt_denorm"].max(),
            best_mae_sample_info["pred_denorm"].max(),
        )
        im_pred_best = axs[0, 2].imshow(
            best_mae_sample_info["pred_denorm"],
            cmap="gray",
            vmin=common_min_best,
            vmax=common_max_best,
        )
        axs[0, 2].set_title(
            f"Najlepšia predikcia\nMAE: {best_mae_sample_info['mae']:.4f}", fontsize=14
        )
        axs[0, 2].axis("off")
        # fig.colorbar(im_pred_best, ax=axs[0, 2], fraction=0.046, pad=0.04)

        # Riadok 2: Najhoršia MAE
        axs[1, 0].imshow(worst_mae_sample_info["input_orig"], cmap="gray")
        axs[1, 0].set_title("Zabalený obraz", fontsize=14)
        axs[1, 0].axis("off")

        im_gt_worst = axs[1, 1].imshow(worst_mae_sample_info["gt_denorm"], cmap="gray")
        axs[1, 1].set_title("Rozbalený referenčný obraz", fontsize=14)
        axs[1, 1].axis("off")
        # fig.colorbar(im_gt_worst, ax=axs[1, 1], fraction=0.046, pad=0.04)

        common_min_worst = min(
            worst_mae_sample_info["gt_denorm"].min(),
            worst_mae_sample_info["pred_denorm"].min(),
        )
        common_max_worst = max(
            worst_mae_sample_info["gt_denorm"].max(),
            worst_mae_sample_info["pred_denorm"].max(),
        )
        im_pred_worst = axs[1, 2].imshow(
            worst_mae_sample_info["pred_denorm"],
            cmap="gray",
            vmin=common_min_worst,
            vmax=common_max_worst,
        )
        axs[1, 2].set_title(
            f"Najhoršia predikcia\nMAE: {worst_mae_sample_info['mae']:.4f}", fontsize=14
        )
        axs[1, 2].axis("off")
        # fig.colorbar(im_pred_worst, ax=axs[1, 2], fraction=0.046, pad=0.04)

        plt.tight_layout()  # Odstránený rect, keďže suptitle už nie je

        run_name_for_title = os.path.splitext(os.path.basename(weights_path))[
            0
        ].replace(
            "best_weights_", ""
        )  # Tento riadok môže zostať, ak ho používate pre názov súboru
        base_save_name = f"best_worst_mae_visualization_{run_name_for_title}"
        save_fig_path_png = f"{base_save_name}.png"
        save_fig_path_svg = f"{base_save_name}.svg"

        plt.savefig(save_fig_path_png)
        print(f"Vizualizácia najlepšej/najhoršej MAE uložená do: {save_fig_path_png}")

        plt.savefig(save_fig_path_svg)
        print(
            f"Vizualizácia najlepšej/najhoršej MAE uložená aj do: {save_fig_path_svg}"
        )

        plt.show()
    else:
        print(
            "Nepodarilo sa nájsť dostatok dát pre vizualizáciu najlepšej/najhoršej MAE."
        )


if __name__ == "__main__":
    # --- NASTAVENIA PRE TESTOVANIE ---
    CONFIG_FILE_PATH = r"C:\Users\viera\Desktop\q_tiff\TRENOVANIE_bakalarka_simul\optimalizacia_hype\trenovanie_5\config_R34imgnet_direct_MAE_GDL0p3_AugMedium_LR5em04_WD1em04_Ep120_ESp30_Tmax120_EtaMin1em07_bs8_bs8.txt"
    WEIGHTS_FILE_PATH = r"C:\Users\viera\Desktop\q_tiff\TRENOVANIE_bakalarka_simul\optimalizacia_hype\trenovanie_5\best_weights_R34imgnet_direct_MAE_GDL0p3_AugMedium_LR5em04_WD1em04_Ep120_ESp30_Tmax120_EtaMin1em07_bs8_bs8.pth"
    TEST_DATA_PATH = r"C:\Users\viera\Desktop\q_tiff\TRENOVANIE_bakalarka_simul\split_dataset_tiff_for_dynamic_v_stratified_final\static_test_dataset"
    # IMAGE_INDEX_TO_VISUALIZE už nie je potrebný pre hlavný graf, ale môžete ho ponechať, ak chcete pridať späť pôvodnú vizualizáciu jedného obrázka.
    DEVICE_TO_USE = "cuda"

    script_start_time = time.time()

    if not os.path.exists(CONFIG_FILE_PATH):
        print(
            f"CHYBA: Konfiguračný súbor '{CONFIG_FILE_PATH}' neexistuje. Skontroluj cestu."
        )
    elif not os.path.exists(WEIGHTS_FILE_PATH):
        print(
            f"CHYBA: Súbor s váhami '{WEIGHTS_FILE_PATH}' neexistuje. Skontroluj cestu."
        )
    else:
        evaluate_and_visualize_model(
            config_path=CONFIG_FILE_PATH,
            weights_path=WEIGHTS_FILE_PATH,
            test_dataset_path=TEST_DATA_PATH,
            # image_index_to_show=IMAGE_INDEX_TO_VISUALIZE, # Odstránené z volania
            device_str=DEVICE_TO_USE,
        )

    script_end_time = time.time()
    total_script_time = script_end_time - script_start_time
    print(
        f"\nCelkový čas vykonávania skriptu: {total_script_time:.2f} sekúnd ({time.strftime('%H:%M:%S', time.gmtime(total_script_time))})."
    )

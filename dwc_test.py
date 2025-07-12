import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
# import torchvision.transforms.v2 as T # Pre testovanie sa augmentácie zvyčajne nepoužívajú
import segmentation_models_pytorch as smp
import tifffile as tiff
import torch
import torch.nn as nn
# import torch.optim as optim # Pre testovanie nie je potrebný optimizer
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# --- Globálne Konštanty (ak sú potrebné) ---
# NORMALIZED_INPUT_CLAMP_MIN = -1.0 # Z dwc.py, ak by bolo potrebné pre nejakú vizualizáciu normalizovaného vstupu
# NORMALIZED_INPUT_CLAMP_MAX = 1.0
KMAX_DEFAULT_FALLBACK = 6  # Fallback, ak K_MAX nie je v configu

# ----------------------------------------------------------------------------------
# KOPÍROVANÉ TRIEDY A FUNKCIE Z dwc.py (alebo ich ekvivalenty)
# ----------------------------------------------------------------------------------


class WrapCountDataset(Dataset):  # Prevzaté a upravené z dwc.py pre testovanie
    def __init__(
        self,
        path_to_data,
        input_min_max_global,
        k_max_val,  # K_MAX sa načíta z configu
        target_img_size=(512, 512),
        edge_loss_weight=1.0,
    ):  # Pre konzistenciu, pri evaluácii sa nepoužije
        self.path = path_to_data
        self.image_list = sorted(glob.glob(os.path.join(self.path, "images", "*.tiff")))
        if not self.image_list:
            print(
                f"VAROVANIE: Nenašli sa žiadne obrázky v {os.path.join(self.path, 'images')}"
            )
        self.input_min_g, self.input_max_g = input_min_max_global
        if self.input_min_g is None or self.input_max_g is None:
            raise ValueError(
                "input_min_max_global musí byť poskytnuté pre WrapCountDataset."
            )
        self.k_max = k_max_val
        self.target_img_size = target_img_size
        self.edge_loss_weight = edge_loss_weight

    def _normalize_input_minmax_to_minus_one_one(self, data, min_val, max_val):
        if max_val == min_val:
            return (
                torch.zeros_like(data)
                if isinstance(data, torch.Tensor)
                else np.zeros_like(data)
            )
        return 2.0 * (data - min_val) / (max_val - min_val) - 1.0

    def _ensure_shape_and_type(
        self, img_numpy, target_shape, data_name="Image", dtype=np.float32
    ):
        img_numpy = img_numpy.astype(dtype)
        current_shape = img_numpy.shape[-2:]  # Funguje pre 2D aj 3D (C,H,W)

        if current_shape != target_shape:
            original_shape_for_debug = img_numpy.shape
            # Ak je menší, padneme
            h, w = current_shape
            target_h, target_w = target_shape

            pad_h = max(0, target_h - h)

            pad_w = max(0, target_w - w)

            if pad_h > 0 or pad_w > 0:
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left

                if img_numpy.ndim == 2:  # (H, W)
                    img_numpy = np.pad(
                        img_numpy,
                        ((pad_top, pad_bottom), (pad_left, pad_right)),
                        mode="reflect",
                    )
                elif img_numpy.ndim == 3 and img_numpy.shape[0] == 1:  # (1, H, W)
                    img_numpy = np.pad(
                        img_numpy,
                        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                        mode="reflect",
                    )
                else:  # Iné tvary (napr. RGB) nie sú očakávané pre tieto dáta
                    print(
                        f"VAROVANIE: Neočakávaný tvar pre padding {data_name}: {img_numpy.shape}. Skúšam ako 2D."
                    )
                    if img_numpy.ndim > 2:
                        img_numpy = (
                            img_numpy.squeeze()
                        )  # Skúsime odstrániť nadbytočné dimenzie
                    if img_numpy.ndim == 2:
                        img_numpy = np.pad(
                            img_numpy,
                            ((pad_top, pad_bottom), (pad_left, pad_right)),
                            mode="reflect",
                        )
                    else:
                        raise ValueError(
                            f"Nepodporovaný tvar pre padding {data_name}: {original_shape_for_debug}"
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
                elif img_numpy.ndim == 3 and img_numpy.shape[0] == 1:  # (1, H, W)
                    img_numpy = img_numpy[
                        :, start_h : start_h + target_h, start_w : start_w + target_w
                    ]
                else:
                    print(
                        f"VAROVANIE: Neočakávaný tvar pre cropping {data_name}: {img_numpy.shape}. Skúšam ako 2D."
                    )
                    if img_numpy.ndim > 2:
                        img_numpy = img_numpy.squeeze()
                    if img_numpy.ndim == 2:
                        img_numpy = img_numpy[
                            start_h : start_h + target_h, start_w : start_w + target_w
                        ]
                    else:
                        raise ValueError(
                            f"Nepodporovaný tvar pre cropping {data_name}: {original_shape_for_debug}"
                        )

            if img_numpy.shape[-2:] != target_shape:
                print(
                    f"VAROVANIE: {data_name} '{getattr(self, 'current_img_path_for_debug', 'N/A')}' mal tvar {original_shape_for_debug}, po úprave na {target_shape} má {img_numpy.shape}. Môže dôjsť k chybe."
                )
        return img_numpy

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        self.current_img_path_for_debug = self.image_list[index]
        img_path = self.image_list[index]
        base_id_name = (
            os.path.basename(img_path).replace("wrappedbg_", "").replace(".tiff", "")
        )
        lbl_path = os.path.join(
            os.path.dirname(os.path.dirname(img_path)),
            "labels",
            f"unwrapped_{base_id_name}.tiff",
        )

        if not os.path.exists(img_path):
            print(f"CHYBA: Vstupný obrázok neexistuje: {img_path}")
            return None, None, None, None, None
        if not os.path.exists(lbl_path):
            print(f"CHYBA: Label obrázok neexistuje: {lbl_path}")
            return None, None, None, None, None

        try:
            wrapped_orig_np = tiff.imread(img_path)
            unwrapped_orig_np = tiff.imread(lbl_path)
        except Exception as e:
            print(f"CHYBA načítania TIFF: {img_path} alebo {lbl_path}. Error: {e}")
            return None, None, None, None, None

        wrapped_orig_np = self._ensure_shape_and_type(
            wrapped_orig_np, self.target_img_size, "Wrapped phase (static_eval)"
        )
        unwrapped_orig_np = self._ensure_shape_and_type(
            unwrapped_orig_np, self.target_img_size, "Unwrapped phase (static_eval)"
        )

        # Normalizovaný vstup pre model
        wrapped_input_norm_np = self._normalize_input_minmax_to_minus_one_one(
            wrapped_orig_np, self.input_min_g, self.input_max_g
        )
        wrapped_input_norm_tensor = torch.from_numpy(
            wrapped_input_norm_np.copy().astype(np.float32)
        ).unsqueeze(
            0
        )  # (1, H, W)

        # k-label pre metriku presnosti
        diff_np = (unwrapped_orig_np - wrapped_orig_np) / (2 * np.pi)
        k_float_np = np.round(diff_np)
        k_float_np = np.clip(k_float_np, -self.k_max, self.k_max)
        k_label_np = (
            k_float_np + self.k_max
        )  # .astype(np.int64) # Pre cross_entropy by mal byť long
        k_label_tensor = torch.from_numpy(k_label_np.copy().astype(np.int64))  # (H,W)

        # Pôvodné dáta pre rekonštrukciu a MAE
        unwrapped_gt_orig_tensor = torch.from_numpy(
            unwrapped_orig_np.copy().astype(np.float32)
        )  # (H,W)
        wrapped_orig_tensor = torch.from_numpy(
            wrapped_orig_np.copy().astype(np.float32)
        )  # (H,W)

        # Weight map sa pri evaluácii zvyčajne nepoužíva, ale pre konzistenciu s collate_fn
        # môžeme vrátiť tensor jednotiek.
        weight_map_tensor = torch.ones_like(k_label_tensor, dtype=torch.float32)

        return (
            wrapped_input_norm_tensor,
            k_label_tensor,
            unwrapped_gt_orig_tensor,
            wrapped_orig_tensor,
            weight_map_tensor,
        )


def collate_fn_skip_none_classification(batch):  # Prevzaté z dwc.py
    # Filter out samples where any of the first 5 elements is None
    batch = list(filter(lambda x: all(item is not None for item in x[:5]), batch))
    if not batch:
        return None, None, None, None, None  # Vráti 5 None hodnôt
    return torch.utils.data.dataloader.default_collate(batch)


def k_label_accuracy_full(logits, klabels):  # Prevzaté z dwc.py
    # logits (B,C,H,W), klabels (B,H,W)
    pred_classes = torch.argmax(logits, dim=1)  # (B,H,W)
    correct = (pred_classes == klabels).float().sum()  # Počet správnych pixelov
    total = klabels.numel()  # Celkový počet pixelov
    if total == 0:
        return torch.tensor(0.0)  # Prípad prázdneho batchu
    return correct / total


# Metriky PSNR a SSIM (vyžaduje skimage) - zostáva z pôvodného
try:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

    SKIMAGE_AVAILABLE = True
except ImportError:
    print("Knižnica scikit-image nie je nainštalovaná. PSNR a SSIM nebudú vypočítané.")
    SKIMAGE_AVAILABLE = False


def calculate_psnr_ssim(gt_img_numpy, pred_img_numpy):  # Zostáva z pôvodného
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

    min_dim = min(gt_img_numpy.shape[-2:])  # Posledné dve dimenzie
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
        except ValueError:  # Môže nastať, ak sú obrázky príliš malé alebo konštantné
            current_ssim = np.nan

    return current_psnr, current_ssim


def evaluate_and_visualize_model(
    config_path, weights_path, test_dataset_path, device_str="cuda"
):
    if not os.path.exists(config_path):
        print(f"CHYBA: Konfiguračný súbor nebol nájdený: {config_path}")
        return
    if not os.path.exists(weights_path):
        print(f"CHYBA: Súbor s váhami nebol nájdený: {weights_path}")
        return

    # --- Načítanie Konfigurácie ---
    config = {}
    with open(config_path, "r") as f:
        for line in f:
            if ":" in line:
                key, value = line.split(":", 1)
                config[key.strip()] = value.strip()

    print("--- Načítaná Konfigurácia Experimentu (Klasifikácia) ---")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("-" * 30)

    encoder_name = config.get("Encoder Name", "resnet34")  # Default z dwc.py
    # K_MAX a NUM_CLASSES
    k_max_val_from_config = config.get("K_MAX")
    if k_max_val_from_config is None:
        print(
            f"VAROVANIE: 'K_MAX' nenájdené v configu, používam fallback: {KMAX_DEFAULT_FALLBACK}"
        )
        k_max_val = KMAX_DEFAULT_FALLBACK
    else:
        k_max_val = int(k_max_val_from_config)

    num_classes_effective_from_config = config.get("NUM_CLASSES")
    if num_classes_effective_from_config is None:
        num_classes_effective = 2 * k_max_val + 1
        print(
            f"VAROVANIE: 'NUM_CLASSES' nenájdené v configu, vypočítavam z K_MAX: {num_classes_effective}"
        )
    else:
        num_classes_effective = int(num_classes_effective_from_config)
        if num_classes_effective != (2 * k_max_val + 1):
            print(
                f"VAROVANIE: Nesúlad medzi NUM_CLASSES ({num_classes_effective}) a K_MAX ({k_max_val}) v configu."
            )
            # Dôverujeme K_MAX pre rekonštrukciu, NUM_CLASSES pre model

    # Input normalization stats
    input_norm_str = config.get("Input Normalization (Global MinMax for Wrapped)")
    global_input_min, global_input_max = None, None
    if input_norm_str:
        try:
            min_str, max_str = input_norm_str.split(",")
            global_input_min = float(min_str.split(":")[1].strip())
            global_input_max = float(max_str.split(":")[1].strip())
            print(
                f"Načítané globálne Min/Max pre vstup: Min={global_input_min:.4f}, Max={global_input_max:.4f}"
            )
        except Exception as e:
            print(
                f"CHYBA pri parsovaní Input Normalization stats: {e}. Normalizácia vstupu nemusí byť správna."
            )
    else:
        print(
            "CHYBA: 'Input Normalization (Global MinMax for Wrapped)' nenájdené v configu."
        )
        return

    if global_input_min is None or global_input_max is None:
        print("CHYBA: Nepodarilo sa načítať normalizačné štatistiky pre vstup. Končím.")
        return

    # --- Príprava Datasetu a DataLoaderu ---
    print(f"\nNačítavam testovací dataset (klasifikačný mód) z: {test_dataset_path}")
    test_dataset = WrapCountDataset(
        path_to_data=test_dataset_path,
        input_min_max_global=(global_input_min, global_input_max),
        k_max_val=k_max_val,
        # target_img_size a edge_loss_weight majú defaulty, ak nie sú kritické pre eval
    )
    if len(test_dataset) == 0:
        print("CHYBA: Testovací dataset je prázdny.")
        return

    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,  # Menší batch_size pre eval
        collate_fn=collate_fn_skip_none_classification,
    )

    # --- Načítanie Modelu ---
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Používam zariadenie: {device}")

    try:
        net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,  # Váhy sa načítajú zo súboru
            in_channels=1,  # Normalizovaný wrapped vstup
            classes=num_classes_effective,  # Počet k-tried
            activation=None,
        ).to(device)
        print(
            f"Používam smp.Unet s enkóderom: {encoder_name}, Počet tried: {num_classes_effective}"
        )
    except Exception as e:
        print(f"CHYBA pri inicializácii smp.Unet: {e}")
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
    print("\nEvaluujem celý testovací dataset (klasifikačný model)...")
    all_mae_reconstructed = []
    all_k_accuracy = []
    all_psnr_reconstructed = []
    all_ssim_reconstructed = []

    best_mae_sample_info = {
        "mae": float("inf"),
        "index": -1,
        "wrapped_orig": None,
        "gt_unwrapped": None,
        "pred_unwrapped_reconstructed": None,
    }
    worst_mae_sample_info = {
        "mae": -1.0,
        "index": -1,
        "wrapped_orig": None,
        "gt_unwrapped": None,
        "pred_unwrapped_reconstructed": None,
    }

    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            if batch_data is None or batch_data[0] is None:
                print(f"Preskakujem chybný batch v teste, iterácia {i}")
                continue

            # wrapped_input_norm_tensor, k_label_tensor, unwrapped_gt_orig_tensor, wrapped_orig_tensor, weight_map_tensor
            (
                input_norm_batch,
                k_labels_gt_batch,
                unwrapped_gt_orig_batch,
                wrapped_orig_batch,
                _,
            ) = batch_data

            input_norm_batch = input_norm_batch.to(device)
            k_labels_gt_batch = k_labels_gt_batch.to(device)  # Pre k-accuracy
            # unwrapped_gt_orig_batch a wrapped_orig_batch zostávajú na CPU pre numpy operácie, alebo presunúť podľa potreby

            logits_batch = net(input_norm_batch)  # (B, NumClasses, H, W)
            pred_classes_batch = torch.argmax(logits_batch, dim=1)  # (B, H, W)

            # Výpočet k-accuracy
            current_k_acc = k_label_accuracy_full(
                logits_batch, k_labels_gt_batch.to(device)
            )  # k_labels musia byť na device
            all_k_accuracy.append(current_k_acc.item())

            # Rekonštrukcia a MAE pre každú vzorku v batchi
            for k_idx in range(pred_classes_batch.size(0)):
                current_sample_global_idx = (
                    i * test_loader.batch_size + k_idx
                )  # Približný index, ak batch_size nie je fixný

                pred_classes_sample = pred_classes_batch[k_idx].cpu()  # (H,W)
                wrapped_orig_sample_numpy = (
                    wrapped_orig_batch[k_idx].cpu().numpy().squeeze()
                )  # (H,W)
                unwrapped_gt_orig_sample_numpy = (
                    unwrapped_gt_orig_batch[k_idx].cpu().numpy().squeeze()
                )  # (H,W)

                k_pred_values_sample = pred_classes_sample.float() - k_max_val  # (H,W)
                unwrapped_pred_reconstructed_numpy = (
                    wrapped_orig_sample_numpy
                    + (2 * np.pi) * k_pred_values_sample.numpy()
                )

                mae = np.mean(
                    np.abs(
                        unwrapped_pred_reconstructed_numpy
                        - unwrapped_gt_orig_sample_numpy
                    )
                )
                all_mae_reconstructed.append(mae)

                if SKIMAGE_AVAILABLE:
                    psnr_val, ssim_val = calculate_psnr_ssim(
                        unwrapped_gt_orig_sample_numpy,
                        unwrapped_pred_reconstructed_numpy,
                    )
                    if not np.isnan(psnr_val):
                        all_psnr_reconstructed.append(psnr_val)
                    if not np.isnan(ssim_val):
                        all_ssim_reconstructed.append(ssim_val)

                # Aktualizácia najlepšej/najhoršej MAE
                if mae < best_mae_sample_info["mae"]:
                    best_mae_sample_info["mae"] = mae
                    best_mae_sample_info["index"] = (
                        current_sample_global_idx  # Ukladáme index vzorky
                    )
                    best_mae_sample_info["wrapped_orig"] = wrapped_orig_sample_numpy
                    best_mae_sample_info["gt_unwrapped"] = (
                        unwrapped_gt_orig_sample_numpy
                    )
                    best_mae_sample_info["pred_unwrapped_reconstructed"] = (
                        unwrapped_pred_reconstructed_numpy
                    )

                if mae > worst_mae_sample_info["mae"]:
                    worst_mae_sample_info["mae"] = mae
                    worst_mae_sample_info["index"] = current_sample_global_idx
                    worst_mae_sample_info["wrapped_orig"] = wrapped_orig_sample_numpy
                    worst_mae_sample_info["gt_unwrapped"] = (
                        unwrapped_gt_orig_sample_numpy
                    )
                    worst_mae_sample_info["pred_unwrapped_reconstructed"] = (
                        unwrapped_pred_reconstructed_numpy
                    )

    avg_mae_rec = np.mean(all_mae_reconstructed) if all_mae_reconstructed else np.nan
    avg_k_acc = np.mean(all_k_accuracy) if all_k_accuracy else np.nan
    avg_psnr_rec = np.mean(all_psnr_reconstructed) if all_psnr_reconstructed else np.nan
    avg_ssim_rec = np.mean(all_ssim_reconstructed) if all_ssim_reconstructed else np.nan

    print("\n--- Celkové Priemerné Metriky na Testovacom Datasete (Klasifikácia) ---")
    print(f"Priemerná MAE (rekonštrukcia): {avg_mae_rec:.4f}")
    print(f"Priemerná k-label Accuracy: {avg_k_acc:.4f}")
    if SKIMAGE_AVAILABLE:
        print(f"Priemerný PSNR (rekonštrukcia): {avg_psnr_rec:.2f} dB")
        print(f"Priemerný SSIM (rekonštrukcia): {avg_ssim_rec:.4f}")

    print("\n--- Extrémne Hodnoty MAE (Rekonštrukcia) ---")
    if best_mae_sample_info["index"] != -1:
        print(
            f"Najlepšia MAE (rekon.): {best_mae_sample_info['mae']:.4f} (index: {best_mae_sample_info['index']})"
        )
    if worst_mae_sample_info["index"] != -1:
        print(
            f"Najhoršia MAE (rekon.): {worst_mae_sample_info['mae']:.4f} (index: {worst_mae_sample_info['index']})"
        )

    # --- Vizualizácia Najlepšej a Najhoršej MAE (Rekonštrukcia) ---
    if best_mae_sample_info["index"] != -1 and worst_mae_sample_info["index"] != -1:
        print(
            f"\nVizualizujem a ukladám najlepší a najhorší MAE prípad (rekonštrukcia)..."
        )

        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        run_name_for_file = os.path.splitext(os.path.basename(weights_path))[0].replace(
            "best_weights_clf_", "eval_clf_"
        )

        # Riadok 1: Najlepšia MAE
        axs[0, 0].imshow(best_mae_sample_info["wrapped_orig"], cmap="gray")
        axs[0, 0].set_title("Zabalený obraz", fontsize=14)
        axs[0, 0].axis("off")

        axs[0, 1].imshow(best_mae_sample_info["gt_unwrapped"], cmap="gray")
        axs[0, 1].set_title("Rozbalený referenčný obraz", fontsize=14)
        axs[0, 1].axis("off")

        common_min_best = min(
            best_mae_sample_info["gt_unwrapped"].min(),
            best_mae_sample_info["pred_unwrapped_reconstructed"].min(),
        )
        common_max_best = max(
            best_mae_sample_info["gt_unwrapped"].max(),
            best_mae_sample_info["pred_unwrapped_reconstructed"].max(),
        )
        axs[0, 2].imshow(
            best_mae_sample_info["pred_unwrapped_reconstructed"],
            cmap="gray",
            vmin=common_min_best,
            vmax=common_max_best,
        )
        axs[0, 2].set_title(
            f"Najlepšia predikcia\nMAE: {best_mae_sample_info['mae']:.4f}", fontsize=14
        )
        axs[0, 2].axis("off")

        # Riadok 2: Najhoršia MAE
        axs[1, 0].imshow(worst_mae_sample_info["wrapped_orig"], cmap="gray")
        axs[1, 0].set_title("Zabalený obraz", fontsize=14)
        axs[1, 0].axis("off")

        axs[1, 1].imshow(worst_mae_sample_info["gt_unwrapped"], cmap="gray")
        axs[1, 1].set_title("Rozbalený referenčný obraz", fontsize=14)
        axs[1, 1].axis("off")

        common_min_worst = min(
            worst_mae_sample_info["gt_unwrapped"].min(),
            worst_mae_sample_info["pred_unwrapped_reconstructed"].min(),
        )
        common_max_worst = max(
            worst_mae_sample_info["gt_unwrapped"].max(),
            worst_mae_sample_info["pred_unwrapped_reconstructed"].max(),
        )
        axs[1, 2].imshow(
            worst_mae_sample_info["pred_unwrapped_reconstructed"],
            cmap="gray",
            vmin=common_min_worst,
            vmax=common_max_worst,
        )
        axs[1, 2].set_title(
            f"Najhoršia predikcia\nMAE: {worst_mae_sample_info['mae']:.4f}", fontsize=14
        )
        axs[1, 2].axis("off")

        plt.tight_layout()
        base_save_name = f"best_worst_mae_reconstruction_{run_name_for_file}"
        save_fig_path_png = f"{base_save_name}.png"
        save_fig_path_svg = f"{base_save_name}.svg"

        plt.savefig(save_fig_path_png)
        print(f"Vizualizácia uložená do: {save_fig_path_png}")
        plt.savefig(save_fig_path_svg)
        print(f"Vizualizácia uložená aj do: {save_fig_path_svg}")
        plt.show()
    else:
        print(
            "Nepodarilo sa nájsť dostatok dát pre vizualizáciu najlepšej/najhoršej MAE."
        )


if __name__ == "__main__":
    # --- NASTAVENIA PRE TESTOVANIE (KLASIFIKÁCIA) ---
    # Tieto cesty musia smerovať na výstupy z klasifikačného tréningu (dwc.py)
    CONFIG_FILE_PATH = r"C:\Users\juraj\Desktop\tradicne_metody\config_clf_R34imgnet_Kmax6_AugMed_LR1e-03_WD1e-04_Ep120_Tmax120_EtaMin1e-07_EdgeW3.0_bs8.txt"  # PRÍKLAD! UPRAV!
    WEIGHTS_FILE_PATH = r"C:\Users\juraj\Desktop\tradicne_metody\best_weights_clf_R34imgnet_Kmax6_AugMed_LR1e-03_WD1e-04_Ep120_Tmax120_EtaMin1e-07_EdgeW3.0_bs8.pth"  # PRÍKLAD! UPRAV!

    TEST_DATA_PATH = r"C:\Users\juraj\Desktop\TRENOVANIE_bakalarka_simul\split_dataset_tiff_for_dynamic_v_stratified_final\static_test_dataset"  # Zostáva rovnaký

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
            device_str=DEVICE_TO_USE,
        )

    script_end_time = time.time()
    total_script_time = script_end_time - script_start_time
    print(
        f"\nCelkový čas vykonávania skriptu: {total_script_time:.2f} sekúnd ({time.strftime('%H:%M:%S', time.gmtime(total_script_time))})."
    )

    print("Hotovo.")
    print("--- Klasifikačný test ukončený --")

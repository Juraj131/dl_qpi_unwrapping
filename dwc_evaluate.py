import os
import glob
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import tifffile as tiff
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

# --- GLOBALNE KONSTANTY ---
KMAX_DEFAULT_FALLBACK = 6

# ----------------------------------------------------------------------------------
# KOPEROVANE TRIEDY A FUNKCIE
# ----------------------------------------------------------------------------------

class WrapCountDataset(Dataset):
    def __init__(self, path_to_data,
                 input_min_max_global,
                 k_max_val,
                 target_img_size=(512,512),
                 edge_loss_weight=1.0):
        # path_to_data: str, cesta k datum
        # input_min_max_global: tuple (float, float), globalne min/max pre vstup
        # k_max_val: int, maximalna hodnota k
        # target_img_size: tuple (int, int), cielova velkost obrazkov
        # edge_loss_weight: float, vaha pre edge loss (pri evaluacii sa nepouziva)
        self.path = path_to_data
        self.image_list = sorted(glob.glob(os.path.join(self.path, 'images', "*.tiff")))
        if not self.image_list:
            print(f"VAROVANIE: Nenasli sa ziadne obrazky v {os.path.join(self.path, 'images')}")
        self.input_min_g, self.input_max_g = input_min_max_global
        if self.input_min_g is None or self.input_max_g is None:
            raise ValueError("input_min_max_global musi byt poskytnute pre WrapCountDataset.")
        self.k_max = k_max_val
        self.target_img_size = target_img_size
        self.edge_loss_weight = edge_loss_weight

    def _normalize_input_minmax_to_minus_one_one(self, data, min_val, max_val):
        # data: np.ndarray alebo torch.Tensor, vstupne data
        # min_val: float, minimalna hodnota pre normalizaciu
        # max_val: float, maximalna hodnota pre normalizaciu
        if max_val == min_val: return torch.zeros_like(data) if isinstance(data, torch.Tensor) else np.zeros_like(data)
        return 2.0 * (data - min_val) / (max_val - min_val) - 1.0

    def _ensure_shape_and_type(self, img_numpy, target_shape, data_name="Image", dtype=np.float32):
        # img_numpy: np.ndarray, obrazok
        # target_shape: tuple (int, int), cielovy tvar (H, W)
        # data_name: str, nazov dat pre logovanie
        # dtype: np.dtype, cielovy datovy typ
        img_numpy = img_numpy.astype(dtype)
        current_shape = img_numpy.shape[-2:]
        
        if current_shape != target_shape:
            original_shape_for_debug = img_numpy.shape
            h, w = current_shape
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
                elif img_numpy.ndim == 3 and img_numpy.shape[0] == 1:
                    img_numpy = np.pad(img_numpy, ((0,0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='reflect')
                else:
                    print(f"VAROVANIE: Neocakavany tvar pre padding {data_name}: {img_numpy.shape}. Skusam ako 2D.")
                    if img_numpy.ndim > 2: img_numpy = img_numpy.squeeze()
                    if img_numpy.ndim == 2:
                         img_numpy = np.pad(img_numpy, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='reflect')
                    else:
                         raise ValueError(f"Nepodporovany tvar pre padding {data_name}: {original_shape_for_debug}")

            h, w = img_numpy.shape[-2:]
            if h > target_h or w > target_w:
                start_h = (h - target_h) // 2
                start_w = (w - target_w) // 2
                if img_numpy.ndim == 2:
                    img_numpy = img_numpy[start_h:start_h+target_h, start_w:start_w+target_w]
                elif img_numpy.ndim == 3 and img_numpy.shape[0] == 1:
                    img_numpy = img_numpy[:, start_h:start_h+target_h, start_w:start_w+target_w]
                else:
                    print(f"VAROVANIE: Neocakavany tvar pre cropping {data_name}: {img_numpy.shape}. Skusam ako 2D.")
                    if img_numpy.ndim > 2: img_numpy = img_numpy.squeeze()
                    if img_numpy.ndim == 2:
                        img_numpy = img_numpy[start_h:start_h+target_h, start_w:start_w+target_w]
                    else:
                        raise ValueError(f"Nepodporovany tvar pre cropping {data_name}: {original_shape_for_debug}")

            if img_numpy.shape[-2:] != target_shape:
                 print(f"VAROVANIE: {data_name} '{getattr(self, 'current_img_path_for_debug', 'N/A')}' mal tvar {original_shape_for_debug}, po uprave na {target_shape} ma {img_numpy.shape}. Moze dojst k chybe.")
        return img_numpy

    def __len__(self): return len(self.image_list)

    def __getitem__(self, index):
        # index: int, index vzorky
        self.current_img_path_for_debug = self.image_list[index]
        img_path = self.image_list[index]
        base_id_name = os.path.basename(img_path).replace('wrappedbg_', '').replace('.tiff','')
        lbl_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'labels', f'unwrapped_{base_id_name}.tiff')
        
        if not os.path.exists(img_path):
            print(f"CHYBA: Vstupny obrazok neexistuje: {img_path}"); return None, None, None, None, None
        if not os.path.exists(lbl_path):
            print(f"CHYBA: Label obrazok neexistuje: {lbl_path}"); return None, None, None, None, None

        try:
            wrapped_orig_np = tiff.imread(img_path)
            unwrapped_orig_np = tiff.imread(lbl_path)
        except Exception as e:
            print(f"CHYBA nacitania TIFF: {img_path} alebo {lbl_path}. Error: {e}")
            return None, None, None, None, None

        wrapped_orig_np = self._ensure_shape_and_type(wrapped_orig_np, self.target_img_size, "Wrapped phase (static_eval)")
        unwrapped_orig_np = self._ensure_shape_and_type(unwrapped_orig_np, self.target_img_size, "Unwrapped phase (static_eval)")

        wrapped_input_norm_np = self._normalize_input_minmax_to_minus_one_one(wrapped_orig_np, self.input_min_g, self.input_max_g)
        wrapped_input_norm_tensor = torch.from_numpy(wrapped_input_norm_np.copy().astype(np.float32)).unsqueeze(0)

        diff_np = (unwrapped_orig_np - wrapped_orig_np) / (2 * np.pi)
        k_float_np = np.round(diff_np)
        k_float_np = np.clip(k_float_np, -self.k_max, self.k_max)
        k_label_np = (k_float_np + self.k_max)
        k_label_tensor = torch.from_numpy(k_label_np.copy().astype(np.int64))
        
        unwrapped_gt_orig_tensor = torch.from_numpy(unwrapped_orig_np.copy().astype(np.float32))
        wrapped_orig_tensor = torch.from_numpy(wrapped_orig_np.copy().astype(np.float32))
        
        weight_map_tensor = torch.ones_like(k_label_tensor, dtype=torch.float32)

        return wrapped_input_norm_tensor, k_label_tensor, unwrapped_gt_orig_tensor, wrapped_orig_tensor, weight_map_tensor

def collate_fn_skip_none_classification(batch):
    # batch: list, zoznam vzoriek z DataLoaderu
    batch = list(filter(lambda x: all(item is not None for item in x[:5]), batch))
    if not batch: return None, None, None, None, None
    return torch.utils.data.dataloader.default_collate(batch)

def k_label_accuracy_full(logits, klabels):
    # logits: torch.Tensor (B,C,H,W), vystup modelu
    # klabels: torch.Tensor (B,H,W), referencne k-labely
    pred_classes = torch.argmax(logits, dim=1)
    correct = (pred_classes == klabels).float().sum()
    total = klabels.numel()
    if total == 0: return torch.tensor(0.0)
    return correct / total

# --- METRIKY KVALITY OBRAZU ---
try:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    print("Kniznica scikit-image nie je nainstalovana. PSNR a SSIM nebudu vypocitane.")
    SKIMAGE_AVAILABLE = False

def calculate_psnr_ssim(gt_img_numpy, pred_img_numpy):
    # gt_img_numpy: np.ndarray, referencny obrazok
    # pred_img_numpy: np.ndarray, predikovany obrazok
    if not SKIMAGE_AVAILABLE:
        return np.nan, np.nan
    
    gt_img_numpy = gt_img_numpy.squeeze()
    pred_img_numpy = pred_img_numpy.squeeze()

    data_range = gt_img_numpy.max() - gt_img_numpy.min()
    if data_range < 1e-6:
        current_psnr = float('inf') if np.allclose(gt_img_numpy, pred_img_numpy) else 0.0
    else:
        current_psnr = psnr(gt_img_numpy, pred_img_numpy, data_range=data_range)

    min_dim = min(gt_img_numpy.shape[-2:])
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

# --- HLAVNA EVALUACNA FUNKCIA ---
def evaluate_and_visualize_model(
    config_path,
    weights_path,
    test_dataset_path,
    device_str='cuda'
    ):
    # config_path: str, cesta ku konfiguracnemu suboru
    # weights_path: str, cesta k suboru s vahami modelu
    # test_dataset_path: str, cesta k adresaru s testovacimi datami
    # device_str: str, zariadenie ('cuda' alebo 'cpu')
    if not os.path.exists(config_path):
        print(f"CHYBA: Konfiguracny subor nebol najdeny: {config_path}")
        return
    if not os.path.exists(weights_path):
        print(f"CHYBA: Subor s vahami nebol najdeny: {weights_path}")
        return

    # --- NACITANIE KONFIGURACIE ---
    config = {}
    with open(config_path, 'r') as f:
        for line in f:
            if ":" in line:
                key, value = line.split(":", 1)
                config[key.strip()] = value.strip()
    
    print("--- Nacitana Konfiguracia Experimentu (Klasifikacia) ---")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("-" * 30)

    encoder_name = config.get("Encoder Name", "resnet34") 
    k_max_val_from_config = config.get("K_MAX")
    if k_max_val_from_config is None:
        print(f"VAROVANIE: 'K_MAX' nenajdene v configu, pouzivam fallback: {KMAX_DEFAULT_FALLBACK}")
        k_max_val = KMAX_DEFAULT_FALLBACK
    else:
        k_max_val = int(k_max_val_from_config)
    
    num_classes_effective_from_config = config.get("NUM_CLASSES")
    if num_classes_effective_from_config is None:
        num_classes_effective = 2 * k_max_val + 1
        print(f"VAROVANIE: 'NUM_CLASSES' nenajdene v configu, vypocitavam z K_MAX: {num_classes_effective}")
    else:
        num_classes_effective = int(num_classes_effective_from_config)
        if num_classes_effective != (2 * k_max_val + 1):
            print(f"VAROVANIE: Nesulad medzi NUM_CLASSES ({num_classes_effective}) a K_MAX ({k_max_val}) v configu.")
            
    input_norm_str = config.get("Input Normalization (Global MinMax for Wrapped)")
    global_input_min, global_input_max = None, None
    if input_norm_str:
        try:
            min_str, max_str = input_norm_str.split(',')
            global_input_min = float(min_str.split(':')[1].strip())
            global_input_max = float(max_str.split(':')[1].strip())
            print(f"Nacitane globalne Min/Max pre vstup: Min={global_input_min:.4f}, Max={global_input_max:.4f}")
        except Exception as e:
            print(f"CHYBA pri parsovani Input Normalization stats: {e}. Normalizacia vstupu nemusi byt spravna.")
    else:
        print("CHYBA: 'Input Normalization (Global MinMax for Wrapped)' nenajdene v configu.")
        return

    if global_input_min is None or global_input_max is None:
        print("CHYBA: Nepodarilo sa nacitat normalizacne statistiky pre vstup. Koncim.")
        return

    # --- PRIPRAVA DATASSETU A DATALOADERU ---
    print(f"\nNacitavam testovaci dataset (klasifikacny mod) z: {test_dataset_path}")
    test_dataset = WrapCountDataset(
        path_to_data=test_dataset_path,
        input_min_max_global=(global_input_min, global_input_max),
        k_max_val=k_max_val
    )
    if len(test_dataset) == 0:
        print("CHYBA: Testovaci dataset je prazdny.")
        return
    
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0, 
                             collate_fn=collate_fn_skip_none_classification)
    
    # --- NACITANIE MODELU ---
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Pouzivam zariadenie: {device}")

    try:
        net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None, 
            in_channels=1,        
            classes=num_classes_effective, 
            activation=None
        ).to(device)
        print(f"Pouzivam smp.Unet s enkoderom: {encoder_name}, Pocet tried: {num_classes_effective}")
    except Exception as e:
        print(f"CHYBA pri inicializacii smp.Unet: {e}")
        return

    try:
        net.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"Vahy modelu uspesne nacitane z: {weights_path}")
    except Exception as e:
        print(f"CHYBA pri nacitani vah modelu: {e}")
        return
    net.eval()

    # --- EVALUACIA CELEHO TESTOVACIEHO SETU ---
    print("\nEvaluujem cely testovaci dataset (klasifikacny model)...")
    all_mae_reconstructed = []
    all_k_accuracy = []
    all_psnr_reconstructed = []
    all_ssim_reconstructed = []
    
    all_pixel_errors_flat_rec = []
    all_samples_data_for_avg_rec = []

    best_mae_sample_info_rec = {"mae": float('inf'), "index": -1, "wrapped_orig": None, "gt_unwrapped": None, "pred_unwrapped_reconstructed": None}
    worst_mae_sample_info_rec = {"mae": -1.0, "index": -1, "wrapped_orig": None, "gt_unwrapped": None, "pred_unwrapped_reconstructed": None}
    avg_mae_sample_info_rec = {"mae": -1.0, "index": -1, "wrapped_orig": None, "gt_unwrapped": None, "pred_unwrapped_reconstructed": None, "diff_from_avg_mae": float('inf')}
    
    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            if batch_data is None or batch_data[0] is None:
                print(f"Preskakujem chybny batch v teste, iteracia {i}")
                continue
            
            input_norm_batch, k_labels_gt_batch, unwrapped_gt_orig_batch, wrapped_orig_batch, _ = batch_data
            
            input_norm_batch = input_norm_batch.to(device)
            k_labels_gt_batch_dev = k_labels_gt_batch.to(device) 

            logits_batch = net(input_norm_batch) 
            pred_classes_batch = torch.argmax(logits_batch, dim=1) 

            current_k_acc = k_label_accuracy_full(logits_batch, k_labels_gt_batch_dev) 
            all_k_accuracy.append(current_k_acc.item())

            for k_idx in range(pred_classes_batch.size(0)):
                current_sample_global_idx = i * test_loader.batch_size + k_idx 
                
                pred_classes_sample = pred_classes_batch[k_idx].cpu() 
                wrapped_orig_sample_numpy = wrapped_orig_batch[k_idx].cpu().numpy().squeeze() 
                unwrapped_gt_orig_sample_numpy = unwrapped_gt_orig_batch[k_idx].cpu().numpy().squeeze() 

                k_pred_values_sample = pred_classes_sample.float() - k_max_val 
                unwrapped_pred_reconstructed_numpy = wrapped_orig_sample_numpy + (2 * np.pi) * k_pred_values_sample.numpy()

                current_pixel_errors_rec = np.abs(unwrapped_pred_reconstructed_numpy - unwrapped_gt_orig_sample_numpy)
                all_pixel_errors_flat_rec.extend(current_pixel_errors_rec.flatten().tolist())

                mae = np.mean(current_pixel_errors_rec)
                all_mae_reconstructed.append(mae)
                
                all_samples_data_for_avg_rec.append((mae, wrapped_orig_sample_numpy, unwrapped_gt_orig_sample_numpy, unwrapped_pred_reconstructed_numpy, current_sample_global_idx))

                if SKIMAGE_AVAILABLE:
                    psnr_val, ssim_val = calculate_psnr_ssim(unwrapped_gt_orig_sample_numpy, unwrapped_pred_reconstructed_numpy)
                    if not np.isnan(psnr_val): all_psnr_reconstructed.append(psnr_val)
                    if not np.isnan(ssim_val): all_ssim_reconstructed.append(ssim_val)
                
                if mae < best_mae_sample_info_rec["mae"]:
                    best_mae_sample_info_rec.update({"mae": mae, "index": current_sample_global_idx, 
                                                 "wrapped_orig": wrapped_orig_sample_numpy, 
                                                 "gt_unwrapped": unwrapped_gt_orig_sample_numpy, 
                                                 "pred_unwrapped_reconstructed": unwrapped_pred_reconstructed_numpy})
                
                if mae > worst_mae_sample_info_rec["mae"]:
                    worst_mae_sample_info_rec.update({"mae": mae, "index": current_sample_global_idx, 
                                                  "wrapped_orig": wrapped_orig_sample_numpy, 
                                                  "gt_unwrapped": unwrapped_gt_orig_sample_numpy, 
                                                  "pred_unwrapped_reconstructed": unwrapped_pred_reconstructed_numpy})

    avg_mae_rec = np.mean(all_mae_reconstructed) if all_mae_reconstructed else np.nan
    avg_k_acc = np.mean(all_k_accuracy) if all_k_accuracy else np.nan
    avg_psnr_rec = np.mean(all_psnr_reconstructed) if all_psnr_reconstructed else np.nan
    avg_ssim_rec = np.mean(all_ssim_reconstructed) if all_ssim_reconstructed else np.nan

    # --- CELKOVE PRIEMERNE METRIKY ---
    print("\n--- Celkove Priemerne Metriky na Testovacom Datasete (Klasifikacia & Rekonstrukcia) ---")
    print(f"Priemerna MAE (rekonstrukcia): {avg_mae_rec:.4f}")
    print(f"Priemerna k-label Accuracy: {avg_k_acc:.4f}")
    if SKIMAGE_AVAILABLE:
        print(f"Priemerny PSNR (rekonstrukcia): {avg_psnr_rec:.2f} dB")
        print(f"Priemerny SSIM (rekonstrukcia): {avg_ssim_rec:.4f}")
    
    if not np.isnan(avg_mae_rec) and all_samples_data_for_avg_rec:
        min_diff_to_avg_mae_rec = float('inf')
        avg_candidate_data_rec = None
        for sample_mae_val, s_wrapped, s_gt_unwrapped, s_pred_unwrapped, s_idx in all_samples_data_for_avg_rec:
            diff = abs(sample_mae_val - avg_mae_rec)
            if diff < min_diff_to_avg_mae_rec:
                min_diff_to_avg_mae_rec = diff
                avg_candidate_data_rec = (sample_mae_val, s_wrapped, s_gt_unwrapped, s_pred_unwrapped, s_idx)
        
        if avg_candidate_data_rec:
            avg_mae_sample_info_rec.update({
                "mae": avg_candidate_data_rec[0], 
                "wrapped_orig": avg_candidate_data_rec[1],
                "gt_unwrapped": avg_candidate_data_rec[2],
                "pred_unwrapped_reconstructed": avg_candidate_data_rec[3],
                "index": avg_candidate_data_rec[4],
                "diff_from_avg_mae": min_diff_to_avg_mae_rec
            })

    # --- EXTREMNE A PRIEMERNE HODNOTY MAE ---
    print("\n--- Extremne a Priemerne Hodnoty MAE (Rekonstrukcia) ---")
    if best_mae_sample_info_rec["index"] != -1:
        print(f"Najlepsia MAE (rekon.): {best_mae_sample_info_rec['mae']:.4f} (index: {best_mae_sample_info_rec['index']})")
    if avg_mae_sample_info_rec["index"] != -1:
        print(f"Vzorka najblizsie k priemernej MAE (rekon. {avg_mae_rec:.4f}): MAE={avg_mae_sample_info_rec['mae']:.4f} (index: {avg_mae_sample_info_rec['index']}, rozdiel: {avg_mae_sample_info_rec['diff_from_avg_mae']:.4f})")
    if worst_mae_sample_info_rec["index"] != -1:
        print(f"Najhorsia MAE (rekon.): {worst_mae_sample_info_rec['mae']:.4f} (index: {worst_mae_sample_info_rec['index']})")

    run_name_for_file = os.path.splitext(os.path.basename(weights_path))[0].replace("best_weights_clf_", "eval_clf_")

    if best_mae_sample_info_rec["index"] != -1 or worst_mae_sample_info_rec["index"] != -1 or avg_mae_sample_info_rec["index"] != -1:
        extreme_mae_log_path_rec = f"extreme_mae_values_reconstruction_{run_name_for_file}.txt"
        with open(extreme_mae_log_path_rec, 'w') as f:
            f.write(f"Experiment (Klasifikacia & Rekonstrukcia): {run_name_for_file}\n")
            f.write("--- Extremne a Priemerne Hodnoty MAE (Rekonstrukcia) ---\n")
            if best_mae_sample_info_rec["index"] != -1:
                f.write(f"Najlepsia MAE (rekon.): {best_mae_sample_info_rec['mae']:.6f} (index: {best_mae_sample_info_rec['index']})\n")
            if avg_mae_sample_info_rec["index"] != -1:
                f.write(f"Vzorka najblizsie k priemernej MAE (rekon. {avg_mae_rec:.6f}): MAE={avg_mae_sample_info_rec['mae']:.6f} (index: {avg_mae_sample_info_rec['index']}, rozdiel: {avg_mae_sample_info_rec['diff_from_avg_mae']:.6f})\n")
            if worst_mae_sample_info_rec["index"] != -1:
                f.write(f"Najhorsia MAE (rekon.): {worst_mae_sample_info_rec['mae']:.6f} (index: {worst_mae_sample_info_rec['index']})\n")
            f.write(f"\nPriemerna MAE (rekonstrukcia, cely dataset): {avg_mae_rec:.6f}\n")
            f.write(f"Priemerna k-label Accuracy (cely dataset): {avg_k_acc:.6f}\n")
        print(f"Extremne a priemerne MAE (rekon.) hodnoty ulozene do: {extreme_mae_log_path_rec}")

    # --- HISTOGRAM CHYB REKONSTRUKCIE ---
    if all_pixel_errors_flat_rec:
        all_pixel_errors_flat_rec_np = np.array(all_pixel_errors_flat_rec)
        plt.figure(figsize=(12, 7))
        plt.hist(all_pixel_errors_flat_rec_np, bins=100, color='mediumseagreen', edgecolor='black', alpha=0.7)
        plt.title('Histogram Absolutnych Chyb Rekonstrukcie (vsetky pixely)', fontsize=16)
        plt.xlabel('Absolutna Chyba Rekonstrukcie (radiany)', fontsize=14)
        plt.ylabel('Pocet Pixelov', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.yscale('log') 
        hist_rec_save_path_png = f"error_histogram_reconstruction_{run_name_for_file}.png"
        hist_rec_save_path_svg = f"error_histogram_reconstruction_{run_name_for_file}.svg"
        plt.savefig(hist_rec_save_path_png)
        plt.savefig(hist_rec_save_path_svg)
        print(f"Histogram chyb rekonstrukcie ulozeny do: {hist_rec_save_path_png} a {hist_rec_save_path_svg}")
        plt.show()
        plt.close()

    # --- VIZUALIZACIA MAE PRIKLADOV ---
    if best_mae_sample_info_rec["index"] != -1 and worst_mae_sample_info_rec["index"] != -1 and avg_mae_sample_info_rec["index"] != -1:
        print(f"\nVizualizujem a ukladam najlepsi, priemerny a najhorsi MAE pripad (rekonstrukcia)...")
        
        samples_to_plot_rec = [
            ("Min MAE (Rekon.)", best_mae_sample_info_rec),
            ("Avg MAE (Rekon.)", avg_mae_sample_info_rec),
            ("Max MAE (Rekon.)", worst_mae_sample_info_rec)
        ]

        all_wrapped_col1 = []
        all_gt_pred_unwrapped_col23 = []

        for _, sample_info in samples_to_plot_rec:
            all_wrapped_col1.append(sample_info["wrapped_orig"])
            all_gt_pred_unwrapped_col23.append(sample_info["gt_unwrapped"])
            all_gt_pred_unwrapped_col23.append(sample_info["pred_unwrapped_reconstructed"])

        vmin_col1_rec = np.min([img.min() for img in all_wrapped_col1 if img is not None]) if any(img is not None for img in all_wrapped_col1) else 0
        vmax_col1_rec = np.max([img.max() for img in all_wrapped_col1 if img is not None]) if any(img is not None for img in all_wrapped_col1) else 1
        if vmax_col1_rec <= vmin_col1_rec: vmax_col1_rec = vmin_col1_rec + 1e-5

        vmin_col23_rec = np.min([img.min() for img in all_gt_pred_unwrapped_col23 if img is not None]) if any(img is not None for img in all_gt_pred_unwrapped_col23) else 0
        vmax_col23_rec = np.max([img.max() for img in all_gt_pred_unwrapped_col23 if img is not None]) if any(img is not None for img in all_gt_pred_unwrapped_col23) else 1
        if vmax_col23_rec <= vmin_col23_rec: vmax_col23_rec = vmin_col23_rec + 1e-5
        
        fig, axs = plt.subplots(3, 4, figsize=(16, 13))

        col_titles_aligned = ["Zabaleny obraz", "Rozbaleny referencny obraz", "Predikcia", "Absolutna chyba"]

        error_map_mappables_rec = []
        img0_for_cbar, img1_for_cbar = None, None

        for i, (row_desc, sample_info) in enumerate(samples_to_plot_rec):
            current_img0 = axs[i, 0].imshow(sample_info["wrapped_orig"], cmap='gray', vmin=vmin_col1_rec, vmax=vmax_col1_rec)
            if i == 0: img0_for_cbar = current_img0

            current_img1 = axs[i, 1].imshow(sample_info["gt_unwrapped"], cmap='gray', vmin=vmin_col23_rec, vmax=vmax_col23_rec)
            if i == 0: img1_for_cbar = current_img1
            
            axs[i, 2].imshow(sample_info["pred_unwrapped_reconstructed"], cmap='gray', vmin=vmin_col23_rec, vmax=vmax_col23_rec)
            
            error_map_rec = np.abs(sample_info["pred_unwrapped_reconstructed"] - sample_info["gt_unwrapped"])
            err_min_val = error_map_rec.min()
            err_max_val = error_map_rec.max()
            if err_max_val <= err_min_val:
                err_max_val = err_min_val + 1e-5
            
            img3 = axs[i, 3].imshow(error_map_rec, cmap='viridis', vmin=err_min_val, vmax=err_max_val)
            error_map_mappables_rec.append(img3)

        for j, col_title_text in enumerate(col_titles_aligned):
            axs[0, j].set_title(col_title_text, fontsize=16, pad=20)

        for ax_row in axs:
            for ax in ax_row:
                ax.axis('off')
        
        for i in range(3):
            fig.colorbar(error_map_mappables_rec[i], ax=axs[i, 3], orientation='vertical', fraction=0.046, pad=0.02, aspect=15)

        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.18, top=0.90, wspace=0.0, hspace=0.06)

        if img0_for_cbar:
            pos_col0_ax2 = axs[2,0].get_position()
            cax1_left = pos_col0_ax2.x0
            cax1_bottom = 0.13 
            cax1_width = pos_col0_ax2.width
            cax1_height = 0.025 
            cax1 = fig.add_axes([cax1_left, cax1_bottom, cax1_width, cax1_height])
            cb1 = fig.colorbar(img0_for_cbar, cax=cax1, orientation='horizontal')

        if img1_for_cbar:
            pos_col1_ax2 = axs[2,1].get_position() 
            pos_col2_ax2 = axs[2,2].get_position() 
            cax23_left = pos_col1_ax2.x0
            cax23_bottom = 0.13 
            cax23_width = (pos_col2_ax2.x0 + pos_col2_ax2.width) - pos_col1_ax2.x0 
            cax23_height = 0.025
            cax23 = fig.add_axes([cax23_left, cax23_bottom, cax23_width, cax23_height])
            cb23 = fig.colorbar(img1_for_cbar, cax=cax23, orientation='horizontal') 
        
        base_save_name_rec = f"detailed_comparison_mae_reconstruction_{run_name_for_file}"
        save_fig_path_png_rec = f"{base_save_name_rec}.png"
        save_fig_path_svg_rec = f"{base_save_name_rec}.svg"

        plt.savefig(save_fig_path_png_rec, dpi=200, bbox_inches='tight') 
        print(f"Detailna vizualizacia (rekonstrukcia) ulozena do: {save_fig_path_png_rec}")
        plt.savefig(save_fig_path_svg_rec, bbox_inches='tight')
        print(f"Detailna vizualizacia (rekonstrukcia) ulozena aj do: {save_fig_path_svg_rec}")
        plt.show()
        plt.close(fig)
    else:
        print("Nepodarilo sa najst dostatok dat pre plnu detailnu vizualizaciu (rekonstrukcia).")

# --- SPUSTENIE SKRIPTU ---
if __name__ == '__main__':
    # --- NASTAVENIA PRE TESTOVANIE ---
    CONFIG_FILE_PATH = r"C:\Users\viera\Desktop\q_tiff\TRENOVANIE_bakalarka_simul\classification\experiment4_hyper\config_clf_R34imgnet_Kmax6_AugMed_LR1em03_WD1em04_Ep120_Tmax120_EtaMin1em07_EdgeW5.0_bs8.txt"
    WEIGHTS_FILE_PATH = r"C:\Users\viera\Desktop\q_tiff\TRENOVANIE_bakalarka_simul\classification\experiment4_hyper\best_weights_clf_R34imgnet_Kmax6_AugMed_LR1em03_WD1em04_Ep120_Tmax120_EtaMin1em07_EdgeW5.0_bs8.pth"
    TEST_DATA_PATH = r'C:\Users\viera\Desktop\q_tiff\TRENOVANIE_bakalarka_simul\split_dataset_tiff_for_dynamic_v_stratified_final\static_test_dataset'
    DEVICE_TO_USE = 'cuda'

    script_start_time = time.time()

    if not os.path.exists(CONFIG_FILE_PATH):
        print(f"CHYBA: Konfiguracny subor '{CONFIG_FILE_PATH}' neexistuje. Skontroluj cestu.")
    elif not os.path.exists(WEIGHTS_FILE_PATH):
        print(f"CHYBA: Subor s vahami '{WEIGHTS_FILE_PATH}' neexistuje. Skontroluj cestu.")
    else:
        evaluate_and_visualize_model(
            config_path=CONFIG_FILE_PATH,
            weights_path=WEIGHTS_FILE_PATH,
            test_dataset_path=TEST_DATA_PATH,
            device_str=DEVICE_TO_USE
        )
    
    script_end_time = time.time()
    total_script_time = script_end_time - script_start_time
    print(f"\nCelkovy cas vykonavania skriptu: {total_script_time:.2f} sekund ({time.strftime('%H:%M:%S', time.gmtime(total_script_time))}).")
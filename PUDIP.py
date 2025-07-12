# -*- coding: utf-8 -*-
"""
Jeden skript pre dávkové PUDIP spracovanie TIFF obrázkov.
Kombinuje PUDIP logiku a dávkové spracovanie.
"""

from __future__ import print_function
import os
import sys
import platform
import glob
import tifffile # Pre prácu s TIFF súbormi
import time
import numpy as np
import torch
import json

# Import pre neurónovú sieť - UISTITE SA, ŽE PRIEČINOK 'models' JE V ROVNAKOM ADRESÁRI
try:
    from models.__init__ import get_net
except ImportError as e:
    print("CHYBA: Nepodarilo sa importovať 'get_net' z priečinka 'models'.")
    print(f"Detail chyby: {e}")
    print("Uistite sa, že priečinok 'models' (z pôvodného PUDIP repozitára) je v rovnakom adresári ako tento skript.")
    sys.exit(1)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Nastavenie pre PyTorch (ak je potrebné)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==============================================================================
# === POMOCNÉ FUNKCIE Z PÔVODNÉHO PUDIP SKRIPTU ===
# ==============================================================================

def get_params(opt_over, net, net_input, downsampler=None):
    opt_over_list = opt_over.split(',')
    params = []
    for opt in opt_over_list:
        if opt == 'net':
            params += [x for x in net.parameters()]
        elif opt == 'down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, f"Neznámy parameter pre optimalizáciu: {opt}"
    return params

def fill_noise(x, noise_type):
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False, f"Neznámy typ šumu: {noise_type}"

def get_noise(input_num, input_depth, method, spatial_size, noise_type='u', var=1./10):
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [input_num, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)
        fill_noise(net_input, noise_type)
        net_input *= var
    elif method == 'meshgrid':
        assert input_depth == 2, "Pre meshgrid musí byť input_depth 2"
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), 
                           np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid_np = np.concatenate([X[None,:], Y[None,:]])
        net_input = np_to_torch(meshgrid_np) # np_to_torch pridá batch dimenziu
    else:
        assert False, f"Neznáma metóda pre generovanie vstupu: {method}"
    return net_input

def np_to_torch(img_np):
    # Pre PUDIP, img_np je často HxW. Chceme BxCxHxW.
    # Ak je img_np 2D (H,W), premeníme na (1,1,H,W)
    if img_np.ndim == 2:
        img_np = img_np[None, None, :, :]
    # Ak je img_np 3D (C,H,W), premeníme na (1,C,H,W)
    elif img_np.ndim == 3:
        img_np = img_np[None, :, :, :]
    # Ak už je 4D (B,C,H,W), necháme tak
    elif img_np.ndim == 4:
        pass
    else:
        raise ValueError(f"Nepodporovaný počet dimenzií pre np_to_torch: {img_np.ndim}")
    return torch.from_numpy(img_np).float() # Uistime sa, že je to float

def torch_to_np(img_var):
    # Vstup je typicky (1,C,H,W). Výstup chceme C,H,W alebo H,W ak C=1.
    return img_var.detach().cpu().squeeze().numpy()

def plot_mat_data(figSize, data, SaveFigPath, FigName):
    # Táto funkcia sa používa, ak SaveRes=True
    if not os.path.exists(SaveFigPath):
        try:
            os.makedirs(SaveFigPath)
        except OSError as e:
            print(f"Chyba pri vytváraní priečinka {SaveFigPath}: {e}")
            return # Nemôžeme uložiť obrázok
            
    fig, ax1 = plt.subplots(1)
    fig.set_figheight(figSize)
    fig.set_figwidth(figSize)
    im1 = ax1.imshow(data)
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, ax=ax1, cax=cax1)
    plt.savefig(os.path.join(SaveFigPath, FigName))
    plt.close(fig)

def unwrap_FD(data):
    dev = data.device
    if len(data.size()) != 4: # Očakáva BxCxHxW
        raise Exception(f'Prosím, preformátujte dáta na správnu dimenziu! Aktuálna: {data.size()}')
    
    dimData = data.size()
    # Horizontálna diferencia
    data_dw = data[:,:,:,1:] - data[:,:,:,:-1]
    # Pridaj nulový stĺpec na koniec pre zachovanie rozmerov
    data_dw_bc = torch.cat([data_dw, torch.zeros((dimData[0], dimData[1], dimData[2], 1), device=dev)], dim=3)
    
    # Vertikálna diferencia
    data_dh = data[:,:,1:,:] - data[:,:,:-1,:]
    # Pridaj nulový riadok na koniec pre zachovanie rozmerov
    data_dh_bc = torch.cat([data_dh, torch.zeros((dimData[0], dimData[1], 1, dimData[3]), device=dev)], dim=2)
    
    # Spoj diferencie pozdĺž dimenzie kanálov (dim=1)
    # Pôvodný kód spájal do (B, 2*C, H, W). Ak C=1, výsledok je (B, 2, H, W).
    data_fd = torch.cat([data_dw_bc, data_dh_bc], dim=1)
    return data_fd

def unwrap_FD_loss(output_fd, data_fd_mod):
    # Jednoduchý rozdiel, strata sa počíta neskôr
    unwrap_fd_residual = output_fd - data_fd_mod
    return unwrap_fd_residual

def wrap_formular(data, constant=2*torch.pi):
    # Modulo operácia pre rozsah (-pi, pi)
    wdata = data - torch.div(data + constant / 2, constant, rounding_mode="floor") * constant
    return wdata

def Plot_Quality(Snr, DataLoss, figSize, SaveFigPath):
    # Táto funkcia sa používa, ak SaveRes=True a target je poskytnutý
    if not os.path.exists(SaveFigPath):
        try:
            os.makedirs(SaveFigPath)
        except OSError as e:
            print(f"Chyba pri vytváraní priečinka {SaveFigPath} pre grafy kvality: {e}")
            return

    fig1, ax1 = plt.subplots()
    fig1.set_figheight(figSize)
    fig1.set_figwidth(figSize)
    line1, = ax1.plot(DataLoss[1:], color='purple', lw=1, ls='-', marker='v', markersize=2, label='DataLoss')
    ax1.legend(loc='best', edgecolor='black', fontsize='x-large')
    ax1.grid(linestyle='dashed', linewidth=0.5)
    plt.title('Loss')
    plt.savefig(os.path.join(SaveFigPath, 'Loss.png'))
    plt.close(fig1)
    
    if Snr is not None and len(Snr) > 1: # Potrebujeme aspoň 2 body pre graf
        fig3, ax3 = plt.subplots()
        fig3.set_figheight(figSize)
        fig3.set_figwidth(figSize)
        plt.plot(Snr[1:]) # Ignoruj prvú hodnotu, ak je to počiatočný stav
        plt.title('SNR')
        plt.savefig(os.path.join(SaveFigPath, 'SNR.png'))
        plt.close(fig3)

def Plot_Image(imag, imagc, maxv, minv, i, figSize, SaveFigPath):
    # Táto funkcia sa používa, ak SaveRes=True
    if not os.path.exists(SaveFigPath):
        try:
            os.makedirs(SaveFigPath)
        except OSError as e:
            print(f"Chyba pri vytváraní priečinka {SaveFigPath} pre obrázky iterácií: {e}")
            return

    fig1 = plt.figure(figsize=(figSize * 2, figSize)) # Upravená veľkosť pre 2 subploty
    plt.subplot(1, 2, 1)
    if maxv == 0 and minv == 0:
        plt.imshow(imag)
    else:
        plt.imshow(imag, vmin=minv, vmax=maxv)
    plt.colorbar()
    plt.title(f'Result_iteration_{i}')
    
    plt.subplot(1, 2, 2)
    if maxv == 0 and minv == 0:
        plt.imshow(imagc)
    else:
        plt.imshow(imagc, vmin=minv, vmax=maxv)
    plt.colorbar()
    plt.title('Congruent solution')
    
    plt.savefig(os.path.join(SaveFigPath, f'Result_iteration_{i}.png'))
    plt.close(fig1)

def SNR(rec, target):
    # Táto funkcia sa používa, ak SaveRes=True a target je poskytnutý
    if torch.is_tensor(rec):
        rec = rec.cpu().detach().numpy()
    if torch.is_tensor(target):
        target = target.cpu().detach().numpy()
    
    if rec.shape != target.shape: # Zjednodušená kontrola
        # Pokus o reshape, ak je jeden z nich (1,H,W) a druhý (H,W)
        rec = np.squeeze(rec)
        target = np.squeeze(target)
        if rec.shape != target.shape:
             raise Exception(f'Rozmery Rec ({rec.shape}) a Target ({target.shape}) sa nezhodujú ani po squeeze!')
    
    # Pre 2D obrázky
    rec_flat = rec.flatten()
    target_flat = target.flatten()
    
    signal_power = np.sum(target_flat**2)
    noise_power = np.sum((rec_flat - target_flat)**2)
    
    if noise_power == 0: # Perfektná rekonštrukcia alebo target je nulový
        return np.inf if signal_power > 0 else 0 # alebo nejaká vysoká hodnota
    
    snr_val = 10 * np.log10(signal_power / noise_power)
    return snr_val

# ==============================================================================
# === HLAVNÁ FUNKCIA PUDIP (mierne upravená) ===
# ==============================================================================

def PUDIP(wrapped_data_np, target_np=None, parserin=None):
    # --- Načítanie a spracovanie parametrov ---
    base_parser = {}
    # Pôvodný PUDIP načítaval 'params_default.json'. Pre tento skript predpokladáme,
    # že 'parserin' (slovník) poskytne všetky potrebné parametre.
    # Ak by si chcel načítať aj defaultný JSON, odkomentuj nasledujúce riadky:
    # default_param_file = 'params_default.json'
    # if os.path.exists(default_param_file):
    #     try:
    #         with open(default_param_file) as f:
    #             base_parser = json.load(f)
    #     except Exception as e:
    #         print(f"Varovanie: Nepodarilo sa načítať '{default_param_file}': {e}")

    if isinstance(parserin, str): # Ak je parserin názov súboru
        try:
            with open(parserin) as f:
                override_parser = json.load(f)
            base_parser.update(override_parser)
        except Exception as e:
            print(f"CHYBA: Nepodarilo sa načítať alebo spracovať parserin súbor '{parserin}': {e}.")
            # V tomto prípade by sme mali vrátiť chybu, ak base_parser nie je dostatočný
            if not base_parser: raise ValueError("Chýbajú parametre.") from e
    elif isinstance(parserin, dict): # Ak je parserin už slovník
        base_parser.update(parserin)
    
    parser = base_parser
    if not parser:
        raise ValueError("Parametre pre PUDIP sú prázdne. Poskytnite platný 'parserin' slovník.")

    # Extrakcia parametrov s .get() pre robustnosť
    FileName = parser.get('FileName', 'generic_pudip_run')
    ImagSize = wrapped_data_np.shape[-2:] # Očakáva sa HxW pre wrapped_data_np

    # Pre PUDIP sa očakáva wrapped_data v tvare (B,C,H,W)
    # np_to_torch to zabezpečí, ak je vstup HxW alebo C,H,W
    wrapped_data = np_to_torch(wrapped_data_np).float() # Uistime sa, že je to float tensor
    
    if target_np is not None:
        target = np_to_torch(target_np).float()
    else:
        target = None

    # Extrahuj parametre zo slovníka `parser`
    RealData = parser.get('RealData', False)
    LR = parser.get('LR', 0.01)
    NoiseType = parser.get('NoiseType', 'Std')
    reg_noise_std = parser.get('reg_noise_std', 0.0)
    input_num = parser.get('input_num', 1) # Pre get_noise, zvyčajne 1
    input_depth = parser.get('input_depth', 128)
    output_depth = parser.get('output_depth', 1)
    num_iter = parser.get('num_iter', 2000)
    reg_loss = parser.get('reg_loss', True)
    update_ite = parser.get('update_ite', 200)
    boundWeights = parser.get('boundWeights', [0.1, 10])
    GDeps = parser.get('GDeps', 1e-8)
    SaveRes = parser.get('SaveRes', False)
    show_every = parser.get('show_every', num_iter + 1) # Aby sa nezobrazovalo počas dávky, ak SaveRes=False
    gpu = parser.get('gpu', True)
    gpuID = parser.get('gpuID', 0)
    main_dir = parser.get('main_dir', '.') # Adresár pre výstupy, ak SaveRes=True
    ItUpOut = parser.get('ItUpOut', 100)
    
    INPUT = parser.get("INPUT", "noise")
    OPTIMIZER = parser.get("OPTIMIZER", "adam")
    OPT_OVER = parser.get("OPT_OVER", "net")
    NET_TYPE = parser.get("NET_TYPE", "skip")
    LR_decrease = parser.get("LR_decrease", False)
    OptiScStepSize = parser.get("OptiScStepSize", 1000)
    OptiScGamma = parser.get("OptiScGamma", 0.5)
    pad = parser.get("pad", "zero")
    # convfilt je použitý vnútri get_net, ak NET_TYPE je napr. 'skip_custom'
    # Pre štandardné typy sietí sa nemusí priamo používať z parsera tu.
    # skip_n33d, skip_n33u, skip_n11, num_scales sú parametre pre get_net
    skip_n33d = parser.get("skip_n33d", 128)
    skip_n33u = parser.get("skip_n33u", 128)
    skip_n11 = parser.get("skip_n11", 4)
    num_scales = parser.get("num_scales", 5)

    figSize = parser.get("figSize", 6)
    upsample_mode = parser.get("upsample_mode", "bilinear")
    act_fun = parser.get("act_fun", "PReLU")
    BatchSize = parser.get("BatchSize", 1) # Malo by byť 1 pre DIP
    bgwin = parser.get("bgwin", [3,52,3,52])

    # --- Príprava cesty pre výsledky (ak SaveRes=True) ---
    ResultPath = "" # Inicializácia
    if SaveRes:
        # ... (konštrukcia ResultFileName a ResultPath ako v pôvodnom PUDIP)
        # Toto je zjednodušené, pôvodný kód mal dlhší reťazec pre ResultFileName
        _tag_real = 'Real_' if RealData else 'Simulate_'
        _tag_fname = str(FileName)
        _tag_net = '_NET_' + str(NET_TYPE)
        _tag_lr = '_LR_' + str(LR)
        _tag_iter = '_Iter_' + str(num_iter)
        ResultFileName = _tag_real + _tag_fname + _tag_net + _tag_lr + _tag_iter
        
        results_root_dir = os.path.join(main_dir, 'PUresults_metadata') # Iný názov, aby sa nemiešal s TIFF výstupmi
        ResultPath = os.path.join(results_root_dir, ResultFileName)
        if not os.path.exists(ResultPath):
            try:
                os.makedirs(ResultPath)
            except Exception as e_mkdir:
                print(f"Varovanie: Nepodarilo sa vytvoriť priečinok {ResultPath}: {e_mkdir}.")
                # Ak sa nepodarí vytvoriť, SaveRes sa pre túto časť de facto vypne
                # SaveRes = False # Radšej necháme, Plot_Image a pod. to ošetria

    # --- Nastavenie zariadenia (GPU/CPU) ---
    if gpu:
        if platform.system() == 'Darwin': # macOS
            if torch.backends.mps.is_available():
                device = torch.device("mps")
                print("Používa sa MPS (Apple Silicon GPU)")
            else:
                device = torch.device("cpu")
                print("MPS nie je k dispozícii, používa sa CPU")
        elif torch.cuda.is_available():
            device = torch.device(f'cuda:{gpuID}')
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            print(f"Používa sa CUDA GPU: {torch.cuda.get_device_name(gpuID)}")
        else:
            device = torch.device("cpu")
            print("CUDA GPU nie je k dispozícii, používa sa CPU")
    else:
        device = torch.device("cpu")
        print("Používa sa CPU ( explicitne nastavené)")
        
    wrapped_data = wrapped_data.to(device)
    if target is not None:
        target = target.to(device)

    # --- Plot merania (ak SaveRes=True) ---
    if SaveRes and wrapped_data.shape[0] == 1 and wrapped_data.shape[1] == 1: # B=1, C=1
        imag_to_plot = torch_to_np(wrapped_data) # Získa HxW numpy array
        plot_mat_data(figSize=figSize, data=imag_to_plot, SaveFigPath=ResultPath, FigName=os.sep+'wrapPhase.png')

    # --- Definícia siete ---
    net = get_net(input_depth=input_depth, NET_TYPE=NET_TYPE,
                  upsample_mode=upsample_mode, pad=pad,
                  n_channels=output_depth, 
                  skip_n33d=skip_n33d, skip_n33u=skip_n33u, 
                  skip_n11=skip_n11, num_scales=num_scales, 
                  act_fun=act_fun).to(device)

    # --- Príprava vstupu pre sieť ---
    net_input = get_noise(input_num=BatchSize, # Malo by byť BatchSize (zvyčajne 1)
                          input_depth=input_depth,
                          method=INPUT, spatial_size=ImagSize,
                          noise_type=NoiseType if reg_noise_std > 0 else 'u', # Typ šumu pre inicializáciu
                          var=1./10).to(device).detach()
    
    net_input_saved = net_input.clone() # Pre prípad regularizácie šumom
    noise_for_reg = net_input.clone()   # Pre generovanie nového šumu v každej iterácii

    # --- Optimalizácia ---
    print(f"Spúšťa sa optimalizácia pre {FileName} ({num_iter} iterácií)...")
    t0 = time.time()
    
    trainable_params = get_params(OPT_OVER, net, net_input)
    optimizer = torch.optim.Adam(trainable_params, lr=LR)
    
    if LR_decrease:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=OptiScStepSize, gamma=OptiScGamma)

    history_snr = [] if target is not None else None
    history_loss = [0.0] # Počiatočná nula pre indexovanie
    
    # Výpočet W(D(wrapped_data)) - konečné diferencie zabalených dát, zabalené
    data_fd = unwrap_FD(wrapped_data) # Výstup je (B, 2*C, H, W)
    data_fd_mod = wrap_formular(data_fd)

    outIte_reg_loss = 0 # Pre aktualizáciu váh pri reg_loss

    for i in range(1, num_iter + 1): # Iterácie od 1 do num_iter
        optimizer.zero_grad()
        
        current_net_input = net_input_saved
        if reg_noise_std > 0 and OPT_OVER.find('input') == -1: # Ak neoptimalizujeme vstup, pridávame šum
            if NoiseType == 'Std': # Použi std z pôvodného šumu
                current_net_input = net_input_saved + (noise_for_reg.detach().std() * reg_noise_std * torch.randn_like(net_input_saved))
            elif NoiseType == 'Normal': # Generuj nový normálny šum
                current_net_input = net_input_saved + (reg_noise_std * torch.randn_like(net_input_saved))
            # 'None' alebo iné varianty znamenajú žiadny pridaný šum v iterácii
        
        output_unwrap = net(current_net_input) # Výstup je (B, output_depth, H, W)
        
        # Odstránenie biasu
        if RealData:
            h_start, h_end = min(bgwin[0], ImagSize[0]-1), min(bgwin[1], ImagSize[0])
            w_start, w_end = min(bgwin[2], ImagSize[1]-1), min(bgwin[3], ImagSize[1])
            if h_start < h_end and w_start < w_end and output_unwrap.numel() > 0: # Kontrola platnosti okna a neprázdneho tenzora
                bias = output_unwrap[0,0].narrow(0,h_start,h_end-h_start).narrow(1,w_start,w_end-w_start).mean().clone()
                output_unwrap = output_unwrap - bias
            else: # Fallback, ak bgwin nie je platné
                output_unwrap = output_unwrap - output_unwrap.min() if output_unwrap.numel() > 0 else output_unwrap
        else:
             output_unwrap = output_unwrap - output_unwrap.min() if output_unwrap.numel() > 0 else output_unwrap # Alebo .mean()
        
        # Výpočet straty
        output_fd = unwrap_FD(output_unwrap) # Výstup je (B, 2*output_depth, H, W)
        unwrap_fd_res = unwrap_FD_loss(output_fd, data_fd_mod) # Rozdiel konečných diferencií
        
        # Pre C=1 (output_depth=1), unwrap_fd_res má 2 kanály (dx, dy)
        # fd_squareloss je (B, H, W)
        fd_squareloss = torch.pow(unwrap_fd_res[:,0,:,:],2) + torch.pow(unwrap_fd_res[:,1,:,:],2)
        fd_squareloss = fd_squareloss.unsqueeze(1) # Pridaj späť kanálovú dimenziu -> (B,1,H,W)
        
        # Numerická stabilita
        fd_sqrt_loss_terms = torch.sqrt(fd_squareloss + GDeps**2) # GDeps**2 pod odmocninou

        # Vážená loss (ak reg_loss=True)
        wn = torch.ones_like(output_unwrap, device=device) # (B,C,H,W)
        if reg_loss:
            if i <= update_ite: # Prvých update_ite iterácií sú váhy 1
                pass # wn je už 1
            # elif i > outIte_reg_loss * update_ite and i <= (outIte_reg_loss + 1) * update_ite +1: # Táto podmienka bola zložitá
            # Zjednodušene: aktualizuj váhy každých update_ite iterácií (po prvej fáze)
            elif (i - 1) % update_ite == 0 and i > update_ite:
                # fd_sqrt_loss_terms je (B,1,H,W), potrebujeme váhy pre každý pixel
                epsn = fd_sqrt_loss_terms.detach().clone() # (B,1,H,W)
                # Váhy sú inverzné k chybe, orezané
                wn_candidate = 1. / torch.clamp(epsn, min=boundWeights[0], max=boundWeights[1])
                # Ak output_unwrap má viac kanálov, možno bude treba replikovať wn_candidate
                if wn.shape[1] == wn_candidate.shape[1]: # Ak C=1
                    wn = wn_candidate
                else: # Ak C > 1, a chceme rovnaké váhy pre všetky výstupné kanály
                    wn = wn_candidate.repeat(1, wn.shape[1], 1, 1)
                outIte_reg_loss +=1 # Počítadlo pre ďalšiu aktualizáciu váh
        
        # Celková strata
        # fd_sqrt_loss_terms je (B,1,H,W), wn je (B,C,H,W). Ak C>1, treba prispôsobiť.
        # Pre output_depth=1, toto je v poriadku.
        data_loss = torch.sum(wn * fd_sqrt_loss_terms)
        
        data_loss.backward()
        optimizer.step()
        
        if LR_decrease:
            scheduler.step()
            
        # Záznam histórie
        history_loss.append(data_loss.item())
        
        # Výpis a ukladanie medzivýsledkov
        if i % ItUpOut == 0:
            log_msg = f'Iter: {i:05d}/{num_iter}  Loss: {data_loss.item():.4f}'
            if target is not None:
                # Rekonštrukcia pre SNR (kongruentná s meraním)
                rec_congruent_np = torch_to_np(output_unwrap + wrap_formular(wrapped_data - output_unwrap))
                current_snr = SNR(rec_congruent_np, torch_to_np(target))
                history_snr.append(current_snr)
                log_msg += f'  SNR: {current_snr:.2f} dB'
            print(log_msg)

        if SaveRes and i % show_every == 0 and i > 0:
            with torch.no_grad():
                out_unwrap_np = torch_to_np(output_unwrap)
                out_congruent_np = torch_to_np(output_unwrap + wrap_formular(wrapped_data - output_unwrap))
                Plot_Image(out_unwrap_np, out_congruent_np, maxv=0, minv=0, i=i, figSize=figSize, SaveFigPath=ResultPath)
                if target is not None: # Plot_Quality potrebuje target pre SNR graf
                     Plot_Quality(Snr=history_snr, DataLoss=history_loss, figSize=figSize, SaveFigPath=ResultPath)
                else: # Plotni len Loss
                     Plot_Quality(Snr=None, DataLoss=history_loss, figSize=figSize, SaveFigPath=ResultPath)


    # --- Finálny výstup ---
    final_output_unwrap = net(net_input_saved if OPT_OVER.find('input') == -1 else net_input) # Použi optimalizovaný vstup, ak bol optimalizovaný

    # Odstránenie biasu z finálneho výstupu
    if RealData:
        h_start, h_end = min(bgwin[0], ImagSize[0]-1), min(bgwin[1], ImagSize[0])
        w_start, w_end = min(bgwin[2], ImagSize[1]-1), min(bgwin[3], ImagSize[1])
        if h_start < h_end and w_start < w_end and final_output_unwrap.numel() > 0:
            bias = final_output_unwrap[0,0].narrow(0,h_start,h_end-h_start).narrow(1,w_start,w_end-w_start).mean().clone()
            final_output_unwrap = final_output_unwrap - bias
        else:
            final_output_unwrap = final_output_unwrap - final_output_unwrap.min() if final_output_unwrap.numel() > 0 else final_output_unwrap
    else:
        final_output_unwrap = final_output_unwrap - final_output_unwrap.min() if final_output_unwrap.numel() > 0 else final_output_unwrap
    
    # Kongruentné riešenie s meraniami
    final_out_congruent = final_output_unwrap + wrap_formular(wrapped_data - final_output_unwrap)
    final_out_np = torch_to_np(final_out_congruent) # Výstup je HxW numpy array

    # Uloženie finálneho TIFF, ak je cesta špecifikovaná
    if 'output_tiff_filepath' in parser and parser['output_tiff_filepath'] is not None:
        try:
            tifffile.imwrite(parser['output_tiff_filepath'], final_out_np.astype(np.float32))
            # Správa o úspešnom uložení sa vypíše v hlavnom skripte
        except Exception as e:
            print(f"CHYBA: Nepodarilo sa uložiť finálny TIFF do {parser['output_tiff_filepath']}: {e}")

    # Uloženie .npy a finálneho obrázka iterácie, ak SaveRes=True
    if SaveRes:
        np.save(os.path.join(ResultPath, 'unwrapped_final.npy'), final_out_np)
        Plot_Image(torch_to_np(final_output_unwrap), final_out_np, maxv=0, minv=0, i=num_iter, figSize=figSize, SaveFigPath=ResultPath)
        if target is not None:
            Plot_Quality(Snr=history_snr, DataLoss=history_loss, figSize=figSize, SaveFigPath=ResultPath)
        else:
            Plot_Quality(Snr=None, DataLoss=history_loss, figSize=figSize, SaveFigPath=ResultPath)

    total_time_seconds = time.time() - t0
    print(f'Celkový čas spracovania pre {FileName}: {total_time_seconds // 60:.0f}m {total_time_seconds % 60:.0f}s')
    
    return final_out_np


# ==============================================================================
# === HLAVNÝ BLOK PRE DÁVKOVÉ SPRACOVANIE ===
# ==============================================================================

if __name__ == "__main__":
    # --- Používateľská konfigurácia ---
    # !!! NASTAVTE TIETO CESTY PODĽA VAŠEJ ŠTRUKTÚRY !!!
    INPUT_DIR = r"C:\Users\viera\Desktop\q_tiff\split_dataset_tiff_for_dynamic_v_stratified_final\static_test_dataset\images"  # Napr. "./data/wrapped_tiffs"
    OUTPUT_DIR = r"C:\Users\viera\Desktop\pudip\unwrapped_teest_pudip" # Napr. "./data/unwrapped_tiffs"
    
    # Skontroluj, či sú cesty zadané (jednoduchá kontrola)
    if INPUT_DIR == "CESTA_K_VSTUPNYM_ZABALENYM_TIFF_OBRAZKOM" or \
       OUTPUT_DIR == "CESTA_K_VYSTUPNYM_ROZBALENYM_TIFF_OBRAZKOM":
        print("CHYBA: Prosím, nastavte premenné INPUT_DIR a OUTPUT_DIR v kóde na správne cesty.")
        sys.exit(1)

    if not os.path.isdir(INPUT_DIR):
        print(f"CHYBA: Vstupný priečinok '{INPUT_DIR}' neexistuje!")
        sys.exit(1)

    USE_GPU = torch.cuda.is_available() or torch.backends.mps.is_available() # Pre NVIDIA aj Apple Silicon
    print(f"Dostupnosť GPU (CUDA/MPS): {USE_GPU}")
    if not USE_GPU:
        print("Varovanie: GPU nie je k dispozícii, spracovanie bude na CPU a môže byť výrazne pomalšie.")

    # --- Parametre pre PUDIP (upravte podľa potreby) ---
    # Tieto parametre budú použité pre každý obrázok.
    # Dynamicky sa nastavia: FileName, output_tiff_filepath, main_dir.
    pudip_base_params = {
        "LR": 0.01,
        "input_num": 1, 
        "input_depth": 64, # Počet kanálov pre vstupný šum
        "output_depth": 1,
        "num_iter": 500, # Znížte pre rýchlejší test, napr. na 200-500
        "NoiseType": "n",
        "reg_noise_std": 0.01, 
        "reg_loss": True,
        "update_ite": 200,
        "boundWeights": [0.1, 10],
        "GDeps": 1e-8,
        
        "skip_n33d": 64, # Parametre pre 'skip' architektúru siete
        "skip_n33u": 64,
        "skip_n11": 4,
        "num_scales": 4,

        "LR_decrease": False,
        "OptiScStepSize": 1000,
        "OptiScGamma": 0.5,
        "act_fun": "PReLU",
        "INPUT": "noise",
        "OPTIMIZER": "adam",
        "OPT_OVER": "net",
        "NET_TYPE": "skip",
        "upsample_mode": "bilinear",
        "pad": "zero",

        "gpu": USE_GPU,
        "gpuID": 0,
        
        "SaveRes": False, # Uložiť metadata (PNG, NPY) do OUTPUT_DIR/PUresults_metadata/...? 
                          # Pre dávkové spracovanie zvyčajne False.
        "RealData": False, # Dôležité: False, ak nemáte špecifickú 'background' oblasť.
        "bgwin": [3, 52, 3, 52], # Ignorované, ak RealData=False.

        "ItUpOut": 200, # Každých koľko iterácií vypísať aktuálnu loss.
        "figSize": 6,
        "BatchSize": 1, # Pre DIP je vždy 1.
    }

    # Vytvor výstupný priečinok, ak neexistuje
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Vytvorený výstupný priečinok: {OUTPUT_DIR}")

    # Nájdi všetky vstupné TIFF súbory
    search_pattern = os.path.join(INPUT_DIR, "wrappedbg_*.tiff")
    tiff_files = sorted(glob.glob(search_pattern))

    if not tiff_files:
        print(f"VAROVANIE: Vo vstupnom priečinku '{INPUT_DIR}' sa nenašli žiadne súbory zodpovedajúce vzoru 'wrappedbg_*.tif'")
        sys.exit()
    
    print(f"Nájdených {len(tiff_files)} TIFF súborov na spracovanie.")
    processed_count = 0
    failed_count = 0

    for tiff_path in tiff_files:
        try:
            print(f"\n--- Spracováva sa súbor: {tiff_path} ---")
            
            base_name_with_ext = os.path.basename(tiff_path)
            base_name_no_ext = os.path.splitext(base_name_with_ext)[0]
            
            identifier = base_name_no_ext.replace("wrappedbg_", "")
            if not identifier: identifier = "unknown" # Fallback
            output_tiff_name = f"unwrapped_{identifier}.tiff"
            full_output_tiff_path = os.path.join(OUTPUT_DIR, output_tiff_name)

            if os.path.exists(full_output_tiff_path):
                print(f"Súbor '{full_output_tiff_path}' už existuje. Preskakujem.")
                processed_count +=1
                continue

            # Načítaj TIFF obrázok
            wrapped_image_data_np = tifffile.imread(tiff_path)
            
            if not isinstance(wrapped_image_data_np, np.ndarray):
                print(f"CHYBA: Nepodarilo sa načítať '{tiff_path}' ako numpy array.")
                failed_count += 1
                continue
            
            if wrapped_image_data_np.dtype != np.float32:
                wrapped_image_data_np = wrapped_image_data_np.astype(np.float32)

            if wrapped_image_data_np.ndim != 2:
                print(f"CHYBA: Obrázok '{base_name_with_ext}' má {wrapped_image_data_np.ndim} dimenzií. Očakávam 2D obrázok. Preskakujem.")
                failed_count += 1
                continue
            
            print(f"Načítaný obrázok: '{base_name_with_ext}', tvar: {wrapped_image_data_np.shape}, typ: {wrapped_image_data_np.dtype}")
            print(f"Rozsah hodnôt: min={wrapped_image_data_np.min():.2f}, max={wrapped_image_data_np.max():.2f} (očakáva sa cca -pi až pi)")

            # Priprav parametre pre PUDIP pre tento konkrétny súbor
            current_run_params = pudip_base_params.copy()
            current_run_params["FileName"] = base_name_no_ext
            current_run_params["output_tiff_filepath"] = full_output_tiff_path
            current_run_params["main_dir"] = OUTPUT_DIR # Ak SaveRes=True, podsúbory pôjdu sem

            print(f"Spúšťam PUDIP pre '{base_name_with_ext}'...")
            
            # Zavolaj PUDIP funkciu
            unwrapped_phase_np = PUDIP(wrapped_data_np=wrapped_image_data_np, 
                                       target_np=None, # Bez učiteľa
                                       parserin=current_run_params)
            
            if os.path.exists(full_output_tiff_path): # Funkcia PUDIP by mala súbor uložiť
                 print(f"ÚSPECH: Rozbalený obrázok uložený do: {full_output_tiff_path}")
                 processed_count += 1
            else:
                 print(f"CHYBA: Zdá sa, že PUDIP neuložil súbor '{full_output_tiff_path}'. Skontrolujte výstup z PUDIP funkcie.")
                 failed_count += 1

        except Exception as e:
            print(f"KRITICKÁ CHYBA pri spracovaní súboru {tiff_path}: {e}")
            import traceback
            traceback.print_exc()
            failed_count += 1
            continue 
    
    print("\n--- Dávkové spracovanie ukončené ---")
    print(f"Celkovo úspešne spracovaných (alebo preskočených): {processed_count}")
    print(f"Celkovo zlyhaných: {failed_count}")
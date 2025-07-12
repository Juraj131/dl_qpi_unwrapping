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
from torchvision.transforms import functional as TF
from torch.nn import init

# ------------------------------------------------
# 1) Kontrola integrity datasetu (images vs labels)
# ------------------------------------------------
def check_dataset_integrity(dataset_path):
    images_dir = os.path.join(dataset_path, 'images')
    labels_dir = os.path.join(dataset_path, 'labels')

    image_files = sorted(glob.glob(os.path.join(images_dir, "*.tiff")))
    label_files = sorted(glob.glob(os.path.join(labels_dir, "*.tiff")))

    for image_file in image_files:
        label_file = image_file.replace('images', 'labels').replace('wrappedbg', 'unwrapped')
        if label_file not in label_files:
            raise FileNotFoundError(f"Label pre obrázok {image_file} nebola nájdená.")
    print(f"Dataset {dataset_path} je v poriadku.")


# ------------------------------------------------
# 2) Dataset s random crop a augmentáciou
# ------------------------------------------------
class CustomDataset(Dataset):
    def __init__(self, path_to_data, crop_size=512, augment=False):
        self.path = path_to_data
        self.crop_size = crop_size
        self.image_list = sorted(glob.glob(os.path.join(self.path, 'images', "*.tiff")))
        self.augment = augment

    def __len__(self):
        return len(self.image_list)

    def random_crop(self, image, label):
        h, w = image.shape[-2:]
        ch, cw = self.crop_size, self.crop_size
        # top = np.random.randint(0, h - ch + 1)
        # left = np.random.randint(0, w - cw + 1)

        top = 0
        left = 0

        cropped_image = image[..., top:top + ch, left:left + cw]
        cropped_label = label[..., top:top + ch, left:left + cw]
        return cropped_image, cropped_label

    def augment_data(self, image, label):
        # Rotácia o 90° násobky
        if np.random.rand() > 0.5:
            k = np.random.randint(0, 4)
            image = torch.rot90(image, k, [1, 2])
            label = torch.rot90(label, k, [1, 2])

        # Horizontálny flip
        if np.random.rand() > 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

        # Vertikálny flip
        if np.random.rand() > 0.5:
            image = TF.vflip(image)
            label = TF.vflip(label)

        return image, label

    def __getitem__(self, index):
        #index = 0 toto je na testovanie
        img_path = self.image_list[index]
        lbl_path = img_path.replace('images', 'labels').replace('wrappedbg', 'unwrapped')

        wrapped = tiff.imread(img_path).astype(np.float32)
        unwrapped = tiff.imread(lbl_path).astype(np.float32)

        wrapped_tensor = torch.tensor(wrapped, dtype=torch.float32).unsqueeze(0)
        unwrapped_tensor = torch.tensor(unwrapped, dtype=torch.float32).unsqueeze(0)

        wrapped_cropped, unwrapped_cropped = self.random_crop(wrapped_tensor, unwrapped_tensor)

        if self.augment:
            wrapped_cropped, unwrapped_cropped = self.augment_data(wrapped_cropped, unwrapped_cropped)

        return wrapped_cropped, unwrapped_cropped



# ------------------------------------------------
# 4) Definícia U-Net
# ------------------------------------------------
class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, filter_size=3, stride=1, pad=1, do_batch=1):
        super().__init__()
        self.do_batch = do_batch
        self.conv = nn.Conv2d(in_size, out_size, filter_size, stride, pad)
        # batchnorm s momentum=0.1
        self.bn = nn.BatchNorm2d(out_size, momentum=0.1)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.do_batch:
            outputs = self.bn(outputs)
        outputs = F.relu(outputs)
        return outputs

class unetConvT2(nn.Module):
    def __init__(self, in_size, out_size, filter_size=3, stride=2, pad=1, out_pad=1):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_size, out_size, filter_size, stride, padding=pad, output_padding=out_pad)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = F.relu(outputs)
        return outputs

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.up = unetConvT2(in_size, out_size)

    def forward(self, inputs1, inputs2):
        inputs2 = self.up(inputs2)
        return torch.cat([inputs1, inputs2], dim=1)

class Unet(nn.Module):
    def __init__(self, filters=(16, 32, 64, 128), in_size=1, out_size=1):
        super().__init__()
        self.filters = filters

        self.conv1 = nn.Sequential(
            unetConv2(in_size, filters[0]),
            unetConv2(filters[0], filters[0])
        )
        self.conv2 = nn.Sequential(
            unetConv2(filters[0], filters[1]),
            unetConv2(filters[1], filters[1])
        )
        self.conv3 = nn.Sequential(
            unetConv2(filters[1], filters[2]),
            unetConv2(filters[2], filters[2])
        )
        self.center = nn.Sequential(
            unetConv2(filters[2], filters[3]),
            unetConv2(filters[3], filters[3])
        )

        self.up3 = unetUp(filters[3], filters[2])
        self.up_conv3 = nn.Sequential(
            unetConv2(filters[2]*2, filters[2]),
            unetConv2(filters[2], filters[2])
        )

        self.up2 = unetUp(filters[2], filters[1])
        self.up_conv2 = nn.Sequential(
            unetConv2(filters[1]*2, filters[1]),
            unetConv2(filters[1], filters[1])
        )

        self.up1 = unetUp(filters[1], filters[0])
        self.up_conv1 = nn.Sequential(
            unetConv2(filters[0]*2, filters[0]),
            unetConv2(filters[0], filters[0])
        )

        self.final = nn.Conv2d(filters[0], out_size, kernel_size=1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        x = F.max_pool2d(conv1, 2)

        conv2 = self.conv2(x)
        x = F.max_pool2d(conv2, 2)

        conv3 = self.conv3(x)
        x = F.max_pool2d(conv3, 2)

        x = self.center(x)

        x = self.up3(conv3, x)
        x = self.up_conv3(x)

        x = self.up2(conv2, x)
        x = self.up_conv2(x)

        x = self.up1(conv1, x)
        x = self.up_conv1(x)

        x = self.final(x)
        return x



# ------------------------------------------------
# 3) Main
# ------------------------------------------------
if __name__ == '__main__':
    check_dataset_integrity('split_dataset_tiff/train_dataset')
    check_dataset_integrity('split_dataset_tiff/valid_dataset')
    check_dataset_integrity('split_dataset_tiff/test_dataset')
 
    # Datasets
    train_dataset = CustomDataset('split_dataset_tiff/train_dataset', crop_size=512, augment=False)
    val_dataset = CustomDataset('split_dataset_tiff/valid_dataset', crop_size=512, augment=False)
    test_dataset = CustomDataset('split_dataset_tiff/test_dataset', crop_size=512, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=4 , shuffle=False, num_workers=0)


    # -----------------------------------------------
    # 5) Strata a metrika
    # ------------------------------------------------
    mse_loss_fn = nn.MSELoss()

    def mae_metric(pred, target):
        return torch.mean(torch.abs(pred - target))

    # ------------------------------------------------
    # 6) Tréning
    # ------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print("Using CUDA:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU.")

    net = Unet(filters=(32, 64, 128, 256), in_size=1, out_size=1).to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=0)

    # -- TU JE ZMENA: StepLR namiesto ReduceLROnPlateau
    #    Každých 100 epôch znížime LR na polovicu.
    milestones = [40, 60, 80]
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=milestones, 
        gamma=0.1
    )

    # Zvýšený počet epôch (ak vyhovuje ~8h limitu)
    num_epochs = 100 # Môžeš zvýšiť, ak nechceš skoro zastaviť

    train_loss_history = []
    val_loss_history = []
    train_mae_history = []
    val_mae_history = []

    best_val_mae = float('inf') # Inicializácia pre sledovanie najlepšej MAE
    best_epoch = -1

    print("Starting training...")

    for epoch in range(num_epochs):
        start_time = time.time()

        # --- TRÉNING ---
        net.train()
        epoch_train_loss = []
        epoch_train_mae = []

        for iter, (data_batch, lbl_batch) in enumerate(train_loader):

            print(f"Epoch {epoch+1}/{num_epochs} | Batch {iter+1}/{len(train_loader)}", end='\r')
            
            data_batch = data_batch.to(device)
            lbl_batch = lbl_batch.to(device)

            optimizer.zero_grad()
            output = net(data_batch)

            loss = mse_loss_fn(output, lbl_batch)
            loss.backward()
            optimizer.step()

            mae_val = mae_metric(output, lbl_batch)
            epoch_train_loss.append(loss.item())
            epoch_train_mae.append(mae_val.item())

        avg_train_loss = np.mean(epoch_train_loss)
        avg_train_mae = np.mean(epoch_train_mae)
        train_loss_history.append(avg_train_loss)
        train_mae_history.append(avg_train_mae)

        # --- VALIDÁCIA ---
        net.eval()
        epoch_val_loss = []
        epoch_val_mae = []
        with torch.no_grad():
            for data_batch, lbl_batch in val_loader:
                data_batch = data_batch.to(device)
                lbl_batch = lbl_batch.to(device)

                output = net(data_batch)
                loss = mse_loss_fn(output, lbl_batch)
                mae_val = mae_metric(output, lbl_batch)

                epoch_val_loss.append(loss.item())
                epoch_val_mae.append(mae_val.item())

        avg_val_loss = np.mean(epoch_val_loss)
        avg_val_mae = np.mean(epoch_val_mae)
        val_loss_history.append(avg_val_loss)
        val_mae_history.append(avg_val_mae)

        # --- LOG + uloženie váh ---
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f}, MAE: {avg_train_mae:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, MAE: {avg_val_mae:.4f}")

        # Ukladanie váh z poslednej epochy (prepíše sa každú epochu)
        torch.save(net.state_dict(), 'last_epoch_weights_unet_tiff.pth')

        # Ukladanie skutočne najlepších váh na základe validačnej MAE
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            best_epoch = epoch + 1
            torch.save(net.state_dict(), 'best_val_mae_weights_unet_tiff.pth')
            print(f"    New best validation MAE: {best_val_mae:.4f} at epoch {best_epoch}. Weights saved to 'best_val_mae_weights_unet_tiff.pth'.")


        # --- STEP SCHEDULER po každej epoche ---
        scheduler.step()

        epoch_duration = time.time() - start_time
        print(f"Epoch time: {epoch_duration:.2f} s")

    # Po ukončení tréningu
    # Váhy z poslednej epochy sú už uložené ako 'last_epoch_weights_unet_tiff.pth'
    # Váhy s najlepšou validačnou MAE sú uložené ako 'best_val_mae_weights_unet_tiff.pth'
    print("Training completed.")
    print(f"Weights from the last epoch saved to 'last_epoch_weights_unet_tiff.pth'.")
    if best_epoch != -1:
        print(f"Best validation MAE weights (MAE: {best_val_mae:.4f} at epoch {best_epoch}) saved to 'best_val_mae_weights_unet_tiff.pth'.")
    else:
        print("No best validation MAE weights were saved (e.g., if training was for 0 epochs or validation MAE never improved).")


    # ------------------------------------------------
    # 7) Testovacia fáza
    # ------------------------------------------------
    # Teraz sa rozhodnite, ktoré váhy chcete použiť na testovanie.
    # Ak chcete testovať na váhach s najlepšou validačnou MAE:
    weights_to_load = 'best_val_mae_weights_unet_tiff.pth'
    if not os.path.exists(weights_to_load): # Fallback na váhy z poslednej epochy, ak by najlepšie neexistovali
        print(f"Warning: '{weights_to_load}' not found. Falling back to 'last_epoch_weights_unet_tiff.pth'.")
        weights_to_load = 'last_epoch_weights_unet_tiff.pth'
    
    print(f"\nLoading weights from '{weights_to_load}' for testing...")
    net.load_state_dict(torch.load(weights_to_load))
    net.eval()

    test_mse = []
    test_mae = []

    print("\nEvaluating on the test set...")
    with torch.no_grad():
        for i, (data_batch, lbl_batch) in enumerate(test_loader):
            data_batch = data_batch.to(device)
            lbl_batch = lbl_batch.to(device)

            output = net(data_batch).squeeze(1)

            for j in range(data_batch.size(0)):
                pred = output[j].cpu().numpy()
                lbl = lbl_batch[j].cpu().numpy()

                mse = np.mean((pred - lbl) ** 2)
                mae = np.mean(np.abs(pred - lbl))
                test_mse.append(mse)
                test_mae.append(mae)

                if i == 0 and j == 0:
                    tiff.imwrite('example_test_output.tiff', pred)
                    plt.figure(figsize=(12, 4))

                    plt.subplot(1, 3, 1)
                    plt.imshow(data_batch[j].cpu().numpy().squeeze(), cmap='gray')
                    plt.title("Vstupný zabalený obraz", fontsize=16)
                    plt.colorbar()

                    plt.subplot(1, 3, 2)
                    lbl = lbl_batch[j].cpu().numpy().squeeze()
                    plt.imshow(lbl, cmap='gray')
                    plt.title("Skutočný rozbalený obraz", fontsize=16)
                    plt.colorbar()

                    plt.subplot(1, 3, 3)
                    pred = output[j].cpu().numpy().squeeze()
                    plt.imshow(pred, cmap='gray')
                    plt.title("Predikovaný rozbalený obraz", fontsize=16)
                    plt.colorbar()

                    plt.tight_layout()
                    plt.savefig('example_visualization.png')
                    plt.close()

    avg_test_mse = np.mean(test_mse)
    avg_test_mae = np.mean(test_mae)
    print("\nTest Results:")
    print(f"  Average Test MSE: {avg_test_mse:.4f}")
    print(f"  Average Test MAE: {avg_test_mae:.4f}")

    # ------------------------------------------------
    # 8) Vykreslenie kriviek
    # ------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Trénovacia MSE strata', linestyle='-', linewidth=2)
    plt.plot(val_loss_history, label='Validačná MSE strata', linestyle='-', linewidth=2) #   ,  marker='o'
    plt.title('Vývoj MSE straty', fontsize=16) # <--- Veľkosť nadpisu je 16
    plt.xlabel('Epocha')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid()
    plt.savefig('mse_loss_curve.png')
    plt.show()
    plt.close()


    plt.figure(figsize=(10, 5))
    plt.plot(train_mae_history, label='Trénovacia MAE', linestyle='-', linewidth=2)
    plt.plot(val_mae_history, label='Validačná MAE', linestyle='-', linewidth=2)
    plt.title('Vývoj MAE', fontsize=16) # <--- Veľkosť nadpisu je 16
    plt.xlabel('Epocha')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid()
    plt.savefig('mae_curve.png')
    plt.show()
    plt.close()
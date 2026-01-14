import torch
from dataset import CRDataset, TileCRDataset        
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import os
import numpy as np
from model_CR import ResNet_CR, VGG_CR, ViT_CR, Resnet_Transformer, ViT_16tiles_CR, Swin_CR, ViT_Global_attn, Vit_Tiles_CR
from loss import mse_coordinate_loss, CR_metrics_coordinates
from utils import EarlyStopper, seed_everything
import wandb
import time

wandb.login()
run = wandb.init(
    project = "Coordinate Regression",
    config={
        "learning_rate": 0.001,
        "architecture": "Vit_with_globalattn",
        "dataset": "Tiles",     
        "batch_size": 16,
        "epochs": 100,
    }
)

seed_everything(42)

# # Data for ViT comprehensive - [b,16,H,W]
# train_data =TileCRDataset(
#         root_dir="/home/ai/test/Tiles/tiles_data/tiles_train",
#         num_patches=16,
#         root_centroids="/home/ai/test/processed_slices_train",
#         use_transform=True
#     )
# train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=2)

# val_data = TileCRDataset(
#         root_dir="/home/ai/test/Tiles/tiles_data/tiles_val",
#         num_patches=16,
#         root_centroids="/home/ai/test/processed_slices_val",
#         use_transform=True
#     )
# val_loader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=2)

train_data = CRDataset(root_dir="/home/ai/test/processed_slices_train", task="coord", transforms=True)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=2)

val_data = CRDataset(root_dir="/home/ai/test/processed_slices_val", task="coord", transforms=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=2)

device = "cuda" if torch.cuda.is_available() else "cpu"

model= ViT_Global_attn(
        img_size=224, 
        patch_size=16, 
        in_channels=1, 
        embed_dim=256, 
        depth=12, 
        num_heads=8, 
        num_landmarks=16, 
        drop_rate=0.1).to(device)

save_dir = "/home/ai/test/Coordinate_Regression/logs/coord_reg_seg1to12"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

early_stopper = EarlyStopper(
    patience=5,
    min_delta=1e-4,
    save_path=os.path.join(save_dir, "vit_with_globalattn.pth"),
)

max_epochs = 100
loss_function = mse_coordinate_loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

log_file = open(f"{save_dir}/vit_with_globalattn.txt", "w")

segments_to_include = list(range(1, 13))  # include only segments 1 to 12
epoch_loss_values = []
val_loss_values = []
euclidean_distances = []
SDR = []

start = time.time()
for epoch in range(max_epochs):
    print(f"{'-'*10} epoch {epoch+1}/{max_epochs} {'-'*10}")

    model.train()
    epoch_loss = 0.0

    for i, data in enumerate(train_loader):
        ct = data["ct_slice"].to(device)          # (B,1,H,W)
        coordinates = data["coordinates"].to(device)    # (B,16,2)
        presence = data["presence"].to(device)    # (B,16)

        optimizer.zero_grad()

        outputs = model(ct)                       # (B,16,2)

        loss = loss_function(outputs, coordinates, presence)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if i % 100 == 0:
            print(f"{i}/{len(train_loader)}  Loss: {loss.item():.4f}")

    scheduler.step()
    epoch_loss /= len(train_loader)

    print(f"Train loss: {epoch_loss:.4f}")

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for data in val_loader:
            ct = data["ct_slice"].to(device)
            coordinates = data["coordinates"].to(device)
            presence = data["presence"].to(device)

            outputs = model(ct)
            loss = loss_function(outputs, coordinates, presence)
            val_loss += loss.item()
            
            metric1, metric2 = CR_metrics_coordinates(outputs, coordinates, presence, segments_to_include=segments_to_include, threshold=10.0, img_size=224)
            euclidean_distances.append(metric1.item())
            SDR.append(metric2.item())
    
    val_loss /= len(val_loader)
    avg_edist = np.mean(euclidean_distances)
    avg_sdr = np.mean(SDR)

    print(f"Validation loss: {val_loss:.4f}, Avg Euclidean Distance: {avg_edist:.2f} pixels, Avg Success Detection Rate: {avg_sdr:.2f}")
    end = time.time()
    msg = f"Epoch {epoch+1}: train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}, Avg Euclidean Distance = {avg_edist:.2f}, Avg Success Detection Rate = {avg_sdr:.2f}, time = {end - start:.2f}\n"
    log_file.write(msg)
    log_file.flush()
    
    wandb.log({
        "train_loss": epoch_loss,
        "val_loss": val_loss,
        "epoch": epoch + 1,
        "Avg_Euclidean_Distance": avg_edist,
        "Avg_Success_Detection_Rate": avg_sdr   
    })

    if early_stopper.early_stop(val_loss, model):
        print("Early stopping triggered")
        break
    
    
    
    












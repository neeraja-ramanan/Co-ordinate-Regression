import torch
import os
import numpy as np
from loss import mse_coordinate_loss, CR_metrics_coordinates
from utils import load_model
from dataset import CRDataset, TileCRDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

test_path = "/home/ai/test/Coordinate_Regression/logs/coord_reg_seg1to12/resnet_sigmoid.pth"
test_data_dir = "/home/ai/test/processed_slices_Test"
batch_size = 16

test_data = CRDataset(root_dir=test_data_dir, task="coord")
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

# test_data = TileCRDataset(
#         root_dir="/home/ai/test/Tiles/tiles_data/tiles_testset",
#         num_patches=16,
#         root_centroids=test_data_dir,
#         use_transform=True)
# test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

model = load_model(test_path, "resnet", device)

model.eval()

total_loss = 0.0
all_metric1 = []
all_metric2 = []

segments_to_include = list(range(1,13)) # Example: include only segments 1 to 12
# Per-landmark trackings
edist_per_landmark_sum = torch.zeros(16, device=device)
presence_per_landmark_sum = torch.zeros(16, device=device)

with torch.no_grad():
    for batch in test_loader:
        ct = batch["ct_slice"].to(device)
        coords_gt = batch["coordinates"].to(device)
        presence = batch["presence"].to(device)
        
        coords_pred = model(ct)
        
        loss = mse_coordinate_loss(coords_pred, coords_gt, presence)
        total_loss += loss.item()
        
        # Overall metrics
        metric1, metric2 = CR_metrics_coordinates(
            coords_pred, coords_gt, presence, segments_to_include=segments_to_include, threshold=10.0, img_size=224
        )
        all_metric1.append(metric1.item())
        all_metric2.append(metric2.item())
        
        # Per-landmark Euclidean distances
        pred_xy = coords_pred * 224
        gt_xy = coords_gt * 224
        diff = pred_xy - gt_xy
        edist = torch.sqrt((diff ** 2).sum(dim=-1))  # (B, 16)
        
        edist_per_landmark_sum += (edist * presence).sum(dim=0)
        presence_per_landmark_sum += presence.sum(dim=0)

# Aggregate results
avg_loss = total_loss / len(test_loader)
avg_edist = np.mean(all_metric1)
avg_sdr = np.mean(all_metric2)
mean_edist_per_landmark = edist_per_landmark_sum / (presence_per_landmark_sum + 1e-6)

# Print results
print(f"Test Loss: {avg_loss:.4f}")
print(f"Average Euclidean Distance: {avg_edist:.2f} pixels")
print(f"Average Success Detection Rate: {avg_sdr:.2%}")
print("\nPer-segment performance:")
for i in range(16):
    count = presence_per_landmark_sum[i].item()
    if count > 0:
        print(f"  Segment {i+1:2d}: {mean_edist_per_landmark[i]:6.2f} pixels (n={count:.0f})")

sample_gt = (coords_gt[0,2,:].cpu().numpy() * 224).astype(int)
sample_pred = (coords_pred[0,2,:].cpu().numpy() * 224).astype(int)  

print("coords_gt:", sample_gt, "coords_pred:", sample_pred)


plt.title("Ct Slice")
plt.imshow(ct[0,0,:,:].cpu().numpy(), cmap='gray')
plt.title("Ground Truth Coordinates")
plt.scatter(sample_gt[0], sample_gt[1], c='r', label='Ground Truth')
plt.title("Predicted Coordinates")
plt.scatter(sample_pred[0], sample_pred[1], c='b', label='Predicted')
plt.legend()
plt.show()


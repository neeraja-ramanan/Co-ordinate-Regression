import torch 
import random
import os
import numpy as np
from scipy.ndimage import center_of_mass
from model_CR import ResNet_CR, VGG_CR, ViT_CR, Resnet_Transformer, ViT_16tiles_CR, Swin_CR, ViT_Global_attn, Vit_Tiles_CR

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def load_model(model_path: str, model: str, device: str):
    if model == "resnet":
        model = ResNet_CR(in_channels=1, num_landmarks=16, pretrained=False).to(device)
    elif model == "vgg19":
        model = VGG_CR(in_channels=1, num_landmarks=16, model="vgg19", pretrained=False).to(device)
    elif model == "vit_patchemb":
        model = ViT_CR(img_size=224,
                num_landmarks=16).to(device)
    elif model == "vit_globalattn":
        model = ViT_Global_attn(img_size=224, patch_size=16, in_channels=1, embed_dim=256, num_landmarks=16).to(device)
    elif model == "resnet_transformer":
        model = Resnet_Transformer(img_size=224,hidden_size=256, num_landmarks=16).to(device)
    elif model == "vit_16tiles":
        model = ViT_16tiles_CR(
            img_size=224,
            in_channels=16,  # 16 organ patches as channels
            embed_dim=256,
            num_landmarks=16
        ).to(device)
    elif model == "vit_comprehensive":
        model = Vit_Tiles_CR(
            patch_hw=224, 
            hidden_size=256, 
            img_size=224, 
            num_organs=16).to(device)
    elif model == "swin":
        model = Swin_CR(
        img_size=224,
        in_chans=1,
        num_landmarks=16,
        model_size='tiny',
        drop_rate=0.1
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def normalize_patch(patch):
    patch = (patch - np.min(patch)) / (np.max(patch) - np.min(patch) + 1e-5)
    return patch.astype(np.float32) 

#easrly stopping based on validation loss
class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0, save_path="best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float('inf')
        self.counter = 0
        self.save_path = save_path

    def early_stop(self, val_loss, model):
        # If improved
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0

            # save model
            torch.save(model.state_dict(), self.save_path)
            print(f"Validation loss improved. Model saved to {self.save_path}")

            return False

        # If NOT improved
        else:
            self.counter += 1
            print(f"No improvement for {self.counter} epochs.")

            if self.counter >= self.patience:
                print("Early stopping triggered.")
                return True

            return False
        
    
# def compute_centroids_from_seg(segmentation, organ_labels):
#     """
#     segmentation: np.ndarray of shape (H, W)
#     organ_labels: list of int labels (length = num_organs)

#     Returns:
#         centroids: dict {label: (x,y) or None if missing}
#     """
#     centroids = {}

#     for label in organ_labels:
#         mask = segmentation == label

#         if mask.sum() == 0:
#             centroids[label] = None  # missing organ
#         else:
#             centroids[label] = center_of_mass(mask)  # (z, y, x)

#     return centroids

# def generate_gaussian_heatmap(shape, center, sigma):
#     """
#     shape: (H, W)
#     center: (x, y)
#     sigma: float

#     Returns:
#         heatmap: np.ndarray (H, W)
#     """
#     xx, yy = np.meshgrid(
#         np.arange(shape[0]),
#         np.arange(shape[1]),
#         indexing="ij"
#     )

#     heatmap = np.exp(
#         -((yy-center[1])**2 + (xx-center[0])**2)
#         / (2 * sigma**2)
#     )

#     return heatmap

# def create_centroid_heatmaps(segmentation, organ_labels, sigma=3):
#     """
#     segmentation: (D, H, W)
#     organ_labels: list[int]
#     sigma: Gaussian std in voxels

#     Returns:
#         heatmaps: (num_organs, D, H, W)
#         presence: (num_organs,) binary mask
#     """
#     X, Y = segmentation.shape
#     heatmaps = np.zeros((len(organ_labels), X, Y), dtype=np.float32)
#     presence = np.zeros(len(organ_labels), dtype=np.float32)

#     centroids = compute_centroids_from_seg(segmentation, organ_labels)
#     print("Computed centroids:", centroids)
#     for i, label in enumerate(organ_labels):
#         if centroids[label] is not None:
#             heatmaps[i] = generate_gaussian_heatmap(
#                 (X, Y), centroids[label], sigma
#             )
#             presence[i] = 1.0
#     print("Generated heatmaps with presence:", presence)
#     return heatmaps, presence 

    
# def heatmap_loss(pred, gt, presence):
#     """
#     pred, gt: (B,16,H,W)
#     presence: (B,16)
#     """
#     loss = (pred - gt) ** 2
#     loss = loss.mean(dim=(2,3))          # (B,16)
#     loss = loss * presence               # mask missing organs
#     return loss.sum() / (presence.sum() + 1e-6)

# def heatmap_mae(pred, gt, presence, fg_weight=10.0):
#     # pred, gt: (B,16,H,W)
#     # presence: (B,16)
#     # weighted sum of MSE loss and Euclidean distance

#     mse = (pred - gt) ** 2

#     fg = gt > 0.01
#     weights = torch.ones_like(gt)
#     weights[fg] = fg_weight

#     loss = mse * weights
#     loss = loss.mean(dim=(2,3))    # (B,16)
#     loss = loss * presence

#     return loss.sum() / (presence.sum() + 1e-6)

# def CR_metrics(pred, gt, presence, threshold=5.0):
#     """
#     pred, gt: (B,16,H,W) heatmaps
#     presence: (B,16)
#     """
#     gt_xy = torch.unravel_index(torch.argmax(gt.view(gt.shape[0], gt.shape[1], -1), dim=2), gt.shape[2:4])  # (B,16), (B,16)
#     pred_xy = torch.unravel_index(torch.argmax(pred.view(pred.shape[0], pred.shape[1], -1), dim=2), pred.shape[2:4])  # (B,16), (B,16)

#     edist = torch.sqrt( (pred_xy[0] - gt_xy[0])**2 + (pred_xy[1] - gt_xy[1])**2 )  # (B,16)
#     edist = edist * presence
    
#     metric1 = edist.sum() / (presence.sum() + 1e-6)
#     success = (edist < threshold).float() * presence  # (B,16)
    
#     metric2 = success.sum() / (presence.sum() + 1e-6) # success detection rate

#     return metric1, metric2



       



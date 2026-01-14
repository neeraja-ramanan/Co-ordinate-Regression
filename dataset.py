import os
import re
import time
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
from torchvision import transforms as T
from utils import normalize_patch

class CRDataset(Dataset):
    def __init__(self, root_dir, transforms=True, task = "coord"):
        self.root_dir = root_dir
        print("Loading dataset from:", root_dir)
        self.samples = sorted(glob.glob(os.path.join(root_dir, "*.npz")))
        print(f"Total samples found: {len(self.samples)}")
        self.transforms = T.Compose([
            T.Lambda(normalize_patch),
            T.ToTensor(),
        ])
        self.task = task

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        data = np.load(path)

        if self.transforms:
            ct = self.transforms(data["ct"])       # (1,H,W)
        else:
            ct = torch.from_numpy(data["ct"]).unsqueeze(0).float()        # (1,H,W)
            
        heatmaps = torch.from_numpy(data["heatmaps"]).float()         # (16,H,W)
        presence = torch.from_numpy(data["presence"]).float()         # (16,)
        segmentation = torch.from_numpy(data["seg"]).long()  # (H,W)

        if self.task == "heatmap":
            sample = {
                "ct_slice": ct,
                "heatmaps": heatmaps,
                "presence": presence,
                "segmentation": segmentation
            }
        
        elif self.task == "coord":
            #coordinates from heatmaps
            coords = []
            for hmap in heatmaps:
                hmap_np = hmap.numpy()
                if hmap_np.sum() == 0:
                    coords.append(torch.tensor([0.0, 0.0]))  # missing organ
                else:
                    y, x = np.unravel_index(np.argmax(hmap_np), hmap_np.shape)
                    #normalize to [0,1]
                    coords.append(torch.tensor([x / hmap_np.shape[1], y / hmap_np.shape[0]]))
                    
            coords = torch.stack(coords).float()  # (16, 2)
            
            sample = {
                "ct_slice": ct,
                "coordinates": coords,
                "presence": presence,
                "segmentation": segmentation
            }

        return sample
    
    
# Dataset for tile-based coordinate regression - 16 organ tile stacked as input    
class TileCRDataset(Dataset):
    def __init__(self, root_dir, num_patches, root_centroids, use_transform = True):
        """
        root_dir: path to tiles_train folder
        num_patches: number of patches per slice (segments)
        """
        self.root_dir = root_dir
        self.root_centroids = root_centroids
        self.num_patches = num_patches
        self.samples = []  
        
        #print(self.samples_centroids[0])
        
        # walk through all CT folders
        for ct_name in sorted(os.listdir(root_dir)):
            ct_path = os.path.join(root_dir, ct_name)

            if not os.path.isdir(ct_path):
                continue
            #print(f"CT folder: {ct_path}")
            # find all slice IDs inside this folder
            files = os.listdir(ct_path)
            slice_ids = set()

            for f in files:
                m = re.match(r"slice_(\d+)_segment_\d+\.npy", f)
                if m:
                    slice_ids.add(m.group(1))
                
            for sid in sorted(slice_ids):
                valid_count = 0
                for i in range(self.num_patches):
                    fname = f"slice_{sid}_segment_{i+1}.npy"
                    fpath = os.path.join(ct_path, fname)

                    if os.path.exists(fpath):
                        patch = np.load(fpath)
                        if patch.ndim == 2 and not np.all(patch == 0):
                            valid_count += 1

                if valid_count >= 5:
                    self.samples.append((ct_path,ct_name,sid))
        
        def normalize_patch(patch):
            patch = (patch - np.min(patch)) / (np.max(patch) - np.min(patch) + 1e-5)
            return patch.astype(np.float32)

        self.use_transform = use_transform
        if use_transform:
            self.transforms = T.Compose([
            T.Lambda(normalize_patch),            
            T.ToTensor(),                           
            T.Resize((224, 224), antialias=True),   
        ])
        else:
            self.transforms = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ct_path, ct_name, slice_id = self.samples[idx]
        patches = []
        mask = []        # 1 = valid patch, 0 = black patch
        
        #print(f"Loading sample: ct name {ct_name}, slice {slice_id}")
        for i in range(self.num_patches):
            filename = f"slice_{slice_id}_segment_{i+1}.npy"
            file_path = os.path.join(ct_path, filename)
            if os.path.exists(file_path):
                patch = np.load(file_path)

                if patch.ndim != 2:
                    raise ValueError(f"Patch not 2D. Got shape {patch.shape} at {file_path}")
                is_black = np.all(patch == 0)

                if self.use_transform:
                    if patch.shape[-1] == 0 or patch.shape[-2] == 0:
                        print("Empty patch path:", file_path)
                        raise ValueError(f"Empty patch shape {patch.shape}")
                    #print(f"patch max and min value: {patch.max()}, {patch.min()}")
                    patch = self.transforms(patch)
                    #print(f"patch max and min value: {patch.max()}, {patch.min()}")
                    patch = torch.squeeze(patch)    # 2D again
            else:
                # Missing patch = treat as black
                patch = torch.zeros((224,224), dtype=torch.float32)
                is_black = True

            # Append patch
            patches.append(patch)

            # Append mask (1 = valid, 0 = black)
            mask.append(0 if is_black else 1)

        patches = torch.stack(patches, dim=0)    # [16, H, W]
        mask = torch.tensor(mask, dtype=torch.long)  # [16]
        
        samples_centroids_path = os.path.join(self.root_centroids, f"{ct_name}_slice{slice_id}.npz")
        #print("Loading centroids from:", samples_centroids_path)
        if os.path.exists(samples_centroids_path):
            data = np.load(samples_centroids_path)
            heatmaps = torch.from_numpy(data["heatmaps"]).float()
            coords = []
            for hmap in heatmaps:
                hmap_np = hmap.numpy()
                if hmap_np.sum() == 0:
                    coords.append(torch.tensor([0.0, 0.0]))  # missing organ
                else:
                    y, x = np.unravel_index(np.argmax(hmap_np), hmap_np.shape) 
                    coords.append(torch.tensor([x / hmap_np.shape[1], y / hmap_np.shape[0]]))   #normalize to [0,1]
                    
            coords = torch.stack(coords).float()  # (16, 2)

        return {
            "ct_slice": patches,
            "presence": mask,
            "coordinates": coords
        }

if __name__ == "__main__":
    start = time.time()
    dataset = TileCRDataset(
        root_dir="/home/ai/test/Tiles/tiles_data/tiles_train",
        num_patches=16,
        root_centroids="/home/ai/test/processed_slices_train",
        use_transform=True
    )
    
    # dataset = CRDataset(
    #     root_dir="/home/ai/test/processed_slices",
    #     transforms=True,
    #     task="coord"
    # )
    
    end = time.time()
    print(f"Dataset loaded in {end - start:.2f} seconds.")
    print("Total samples in dataset:", len(dataset))
    
    start = time.time()
    sample = dataset[100]
    print("CT slice shape:", sample['ct_slice'].shape)
    print("Coordinates:", sample['coordinates'])
    print("Presence:", sample['presence'])

    validate_coords = sample['coordinates']* sample['presence'].unsqueeze(1)
    print("Validated Coordinates:", validate_coords)

    
# print("test")
# from torch.utils.data import Dataset
# import time
# import os
# import nibabel as nib
# import numpy as np
# from heatmap import create_centroid_heatmaps, resize_segment
# import torch

# class CRDataset(Dataset):
#     def __init__(
#         self,
#         root_dir_ct,
#         root_dir_seg,
#         organ_labels=list(range(1, 17)),
#         sigma=5,
#         target_shape=(256, 256),
#         transform=None
#     ):
#         self.organ_labels = organ_labels
#         self.sigma = sigma
#         self.target_shape = target_shape
#         self.transform = transform

#         print("Loading dataset...")
#         ct_files = sorted([f for f in os.listdir(root_dir_ct) if f.endswith(".nii.gz")])
#         seg_files = sorted([f for f in os.listdir(root_dir_seg) if f.endswith(".nii.gz")])

#         assert len(ct_files) == len(seg_files), "CT and Seg file count mismatch"

#         self.samples = []  # (ct_path, seg_path, z)

#         for ct_f, seg_f in zip(ct_files, seg_files):
#             ct_path = os.path.join(root_dir_ct, ct_f)
#             seg_path = os.path.join(root_dir_seg, seg_f)

#             seg = nib.load(seg_path).get_fdata()  # seg only (lighter than CT)

#             for z in range(seg.shape[2]):
#                 seg_slice = seg[:, :, z]
#                 unique = np.unique(seg_slice)
#                 num_organs = sum(lbl in self.organ_labels for lbl in unique)

#                 if num_organs >= 5:
#                     self.samples.append((ct_path, seg_path, z))
#         print(f"Total samples (slices with >=5 organs): {len(self.samples)}")
        
#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         print(f"Fetching sample {idx}/{len(self.samples)}")
#         ct_path, seg_path, z = self.samples[idx]
#         print(f"loading sample idx={idx}: CT={ct_path}, SEG={seg_path}, slice={z}")
#         ct = nib.load(ct_path).get_fdata()
#         seg = nib.load(seg_path).get_fdata()
#         assert ct.shape == seg.shape, "CT and Seg shape mismatch"
#         ct_slice = ct[:, :, z]
#         seg_slice = seg[:, :, z]

#         ct_slice = resize_segment(ct_slice, self.target_shape)
#         seg_slice = resize_segment(seg_slice, self.target_shape)

#         heatmaps, presence = create_centroid_heatmaps(
#             seg_slice, self.organ_labels, self.sigma
#         )
#         ct = torch.from_numpy(ct_slice).unsqueeze(0).float()        # (1,H,W)
#         heatmaps = torch.from_numpy(heatmaps).float()         # (16,H,W)
#         presence = torch.from_numpy(presence).float()         # (16,)
#         return {
#             "ct_slice": ct_slice,
#             "segmentation": seg_slice,
#             "heatmaps": heatmaps,
#             "presence": presence,
#         }

    
# if __name__ == "__main__":
#     start = time.time()
#     dataset = CRDataset(
#         root_dir_ct = "/home/ai/Downloads/WORD-V0.1.0/imagesTr",
#         root_dir_seg = "/home/ai/Downloads/WORD-V0.1.0/labelsTr"
#     )
#     end = time.time()
#     print(f"Dataset loaded in {end - start:.2f} seconds.")
    
#     start = time.time()
#     sample = dataset[0]
#     print("CT slice shape:", sample['ct_slice'].shape)
#     print("Segmentation shape:", sample['segmentation'].shape)
#     print("Heatmaps shape:", sample['heatmaps'].shape)
#     print("Presence:", sample['presence'])
    
#     sample2 = dataset[1]
#     sample3 = dataset[2]
#     end = time.time()
#     print(f"3 samples loaded in {end - start:.2f} seconds.")

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import time
import glob
import matplotlib.pyplot as plt
from torchvision import transforms as T
from heatmap import normalize_patch

class CRDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        print("Loading dataset from:", root_dir)
        self.samples = sorted(glob.glob(os.path.join(root_dir, "*.npz")))
        print(f"Total samples found: {len(self.samples)}")
        self.transforms = T.Compose([
            T.Lambda(normalize_patch),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        data = np.load(path)

        if self.transforms:
            ct = self.transforms(data["ct"]).unsqueeze(0)        # (1,H,W)
        else:
            ct = torch.from_numpy(data["ct"]).unsqueeze(0).float()        # (1,H,W)
            
        heatmaps = torch.from_numpy(data["heatmaps"]).float()         # (16,H,W)
        presence = torch.from_numpy(data["presence"]).float()         # (16,)
        segmentation = torch.from_numpy(data["seg"]).long()  # (H,W)

        sample = {
            "ct_slice": ct,
            "heatmaps": heatmaps,
            "presence": presence,
            "segmentation": segmentation
        }

        return sample
 
if __name__ == "__main__":
    start = time.time()
    dataset = CRDataset(
        root_dir="/home/ai/test/processed_slices"
    )
    end = time.time()
    print(f"Dataset loaded in {end - start:.2f} seconds.")
    
    start = time.time()
    sample = dataset[0]
    print("CT slice shape:", sample['ct_slice'].shape)
    print("Heatmaps shape:", sample['heatmaps'].shape)
    print("Presence:", sample['presence'])
    
    sample2 = dataset[1]
    sample3 = dataset[2]
    end = time.time()
    print(f"3 samples loaded in {end - start:.2f} seconds.")
    
    plt.figure(figsize=(12, 6))
    plt.imshow(sample["ct_slice"].squeeze(), cmap='gray', origin='lower')
    # plt.imshow(sample["heatmaps"][8,:,:], cmap='hot', alpha=0.3, origin='lower')  
    # plt.imshow(sample["segmentation"], cmap='hot', alpha=0.5, origin='lower')
    plt.show()
    
    
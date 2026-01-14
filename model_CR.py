import torch
import torch.nn as nn
from torchvision.models import resnet18, vgg16, vgg19
from monai.networks.nets.swin_unetr import SwinTransformer
from monai.networks.blocks import PatchEmbeddingBlock

#Resnet Coordinate Regression Model
class ResNet_CR(nn.Module):
    def __init__(self, in_channels=1, num_landmarks=16, pretrained=False):
        super().__init__()

        backbone = resnet18(weights=None)

        # Replace first conv for 1-channel input
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        if pretrained:
            self.conv1.weight.data = backbone.conv1.weight.data.mean(dim=1, keepdim=True)

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1  # 64
        self.layer2 = backbone.layer2  # 128
        self.layer3 = backbone.layer3  # 256
        self.layer4 = backbone.layer4  # 512
        self.avgpool = backbone.avgpool


        self.coord_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_landmarks * 2)
        )

    def forward(self, x):
        # Encoder
        B, _, H, W = x.shape
        x = self.conv1(x)      # (B,64,112,112)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)    # (B,64,56,56)

        x = self.layer1(x)     # (B,64,56,56)
        x = self.layer2(x)     # (B,128,28,28)
        x = self.layer3(x)     # (B,256,14,14)
        x = self.layer4(x)     # (B,512,7,7)
        x = self.avgpool(x)    # (B,512,1,1)
        x = torch.flatten(x, 1)  # (B,512)
        # Decoder
        x = self.coord_head(x)    # (B,32,224,224)
        coords = x.reshape(B, 16, 2)
        coords = torch.sigmoid(coords)                     # normalize to [0, 1]

        return coords

# VGG Coordinate Regression Model
class VGG_CR(nn.Module):
    def __init__(self, in_channels=1, num_landmarks=16, model="vgg19", pretrained=False):
        super().__init__()
        if model == "vgg19":
            backbone = vgg19(weights=None)
        else:
            backbone = vgg16(weights=None)

        # Use VGG feature extractor
        self.features = backbone.features

        # Replace first conv for 1-channel input
        self.features[0] = nn.Conv2d(
            in_channels, 64, kernel_size=3, padding=1
        )

        self.avgpool = backbone.avgpool  # AdaptiveAvgPool2d((7,7))

        self.coord_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(256, num_landmarks * 2)
        )

    def forward(self, x):
        B = x.shape[0]

        x = self.features(x)        # (B, 512, 7, 7)
        x = self.avgpool(x)         # (B, 512, 7, 7)
        x = torch.flatten(x, 1)     # (B, 512*7*7)

        x = self.coord_head(x)      # (B, 32)
        x = torch.sigmoid(x)        # normalize to [0, 1]

        coords = x.view(B, 16, 2)   # (B, 16, 2)
        return coords


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, embed_dim=256, patch_size=16):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                              kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=256, depth=12, num_heads=8, mlp_ratio=4.0, drop_rate=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=drop_rate,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            for _ in range(depth)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# ViT Coordinate Regression Model
class ViT_CR(nn.Module):
    """
    ViT-based model for direct coordinate regression.
    Outputs (B, 16, 2) - x,y coordinates for 16 landmarks
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=1,
        embed_dim=256,
        depth=12,
        num_heads=8,
        num_landmarks=16,
        drop_rate=0.1
    ):
        super().__init__()

        assert img_size % patch_size == 0, "Image size must be divisible by patch size"

        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.num_landmarks = num_landmarks
        self.img_size = img_size

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer encoder
        self.transformer = TransformerEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            drop_rate=drop_rate
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Coordinate regression head
        self.coord_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, num_landmarks * 2)  # 16 landmarks * 2 coords
        )

    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W) input CT slices
        Returns:
            coords: (B, num_landmarks, 2) normalized coordinates in [0, 1]
        """
        B, _, H, W = x.shape

        # Patch embedding
        x = self.patch_embed(x)  # (B, N, D)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer
        x = self.transformer(x)  # (B, N, D)
        x = self.norm(x)

        # Global average pooling across patches
        x = x.mean(dim=1)  # (B, D)

        # Coordinate prediction
        coords = self.coord_head(x)  # (B, num_landmarks * 2)
        coords = coords.reshape(B, self.num_landmarks, 2)  # (B, 16, 2)

        # Normalize to [0, 1] range using sigmoid
        coords = torch.sigmoid(coords)

        return coords



# ResNet + Transformer Coordinate Regression Model
class Resnet_Transformer(nn.Module):
    """
    ResNet backbone + Transformer for coordinate regression.
    More similar to your original heatmap model structure.
    """
    def __init__(
        self,
        hidden_size=256,
        num_landmarks=16,
        in_channels=1,
        img_size=224,
        num_transformer_layers=6
    ):
        super().__init__()
        
        self.num_landmarks = num_landmarks
        self.img_size = img_size
        
        # ResNet backbone
        backbone = resnet18(weights=None)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # 64
        self.layer2 = backbone.layer2  # 128
        self.layer3 = backbone.layer3  # 256
        self.layer4 = backbone.layer4  # 512
        
        # Project ResNet features to transformer dimension
        self.proj = nn.Sequential(
            nn.Linear(512, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.norm = nn.LayerNorm(hidden_size)
        
        # Coordinate regression head
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_landmarks * 2)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W)
        Returns:
            coords: (B, num_landmarks, 2) normalized coordinates
        """
        B, _, H, W = x.shape

        # ResNet feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # (B, 512, Hf, Wf)
        
        B, C, Hf, Wf = x.shape

        # Flatten spatial map into tokens
        feats = x.flatten(2).transpose(1, 2)  # (B, N=Hf*Wf, C=512)

        # Project to transformer dimension
        feats = self.proj(feats)  # (B, N, hidden_size)

        # Transformer encoder
        feats = self.transformer(feats)
        feats = self.norm(feats)

        # Global pooling
        feats = feats.mean(dim=1)  # (B, hidden_size)

        # Coordinate prediction
        coords = self.coord_head(feats)  # (B, num_landmarks * 2)
        coords = coords.reshape(B, self.num_landmarks, 2)
        
        # Normalize to [0, 1]
        coords = torch.sigmoid(coords)
        
        return coords
    
# ViT 16 patches input 
"""
ViT for Coordinate Regression with 16 Organ Patches as Input Channels (B, 16, 224, 224)
"""

class PatchEmbedding_16(nn.Module):
    def __init__(self, in_channels=16, embed_dim=256, patch_size=16):
        super().__init__()
        # Conv2d will process all 16 channels together
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                              kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class ViT_16tiles_CR(nn.Module):
    """
    ViT-based model for direct coordinate regression.
    Takes 16 patches (of organs) as input
    16 patches are considered as different channels 
    Outputs (B, 16, 2) - x,y coordinates for 16 landmarks - coordinates from CT slice
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=16,  # 16 organ patches as channels
        embed_dim=256,
        depth=12,
        num_heads=8,
        num_landmarks=16,
        drop_rate=0.1
    ):
        super().__init__()

        assert img_size % patch_size == 0, "Image size must be divisible by patch size"

        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 14*14 = 196
        self.num_landmarks = num_landmarks
        self.img_size = img_size
        self.in_channels = in_channels

        # Patch embedding - processes all 16 channels together
        self.patch_embed = PatchEmbedding_16(
            in_channels=in_channels,  # 16 channels
            embed_dim=embed_dim,
            patch_size=patch_size
        )

        # Positional embedding - same as before
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer encoder
        self.transformer = TransformerEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            drop_rate=drop_rate
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Coordinate regression head
        self.coord_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, num_landmarks * 2)  # 16 * 2 coords
        )
    
    def forward(self, x):
        """
            x: (B, 16, H, W) - 16 organ patches stacked as channel
            coords: (B, 16, 2) - normalized coordinates in [0, 1]
        """
        B, P, H, W = x.shape
        
        # Verify input has 16 channels
        assert P == self.in_channels, f"Expected {self.in_channels} channels, got {P}"
    
        x = self.patch_embed(x)  # (B, 196, 256)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.transformer(x) # (B, 196, 256)
        x = self.norm(x)

        x = x.mean(dim=1) # (B, 256)
        
        # Coordinate prediction
        # Input:  (B, 256)
        # Output: (B, 32) -> (B, 16, 2)
        coords = self.coord_head(x)  # (B, num_landmarks * 2)
        coords = coords.reshape(B, self.num_landmarks, 2)  # (B, 16, 2)

        coords = torch.sigmoid(coords)

        return coords
    

class Vit_Tiles_CR(nn.Module):
    """
    Vit_tiles model used for permutation prediction adapted for Coordinate Regression

    Input:  (B, 16, 224, 224)
    Output: (B, 16, 2)
    Each organ tile is treated as one token (like permutation model)
    """

    def __init__(
        self,
        patch_hw=224,
        hidden_size=256,
        img_size=224,
        num_organs=16,
        num_layers=6,
        num_heads=8,
        drop_rate=0.1,
    ):
        super().__init__()

        self.num_organs = num_organs
        self.hidden_size = hidden_size

        # Per-organ patch embedding (one token per organ)
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=1,
            img_size=(img_size, img_size),
            patch_size=(patch_hw, patch_hw),
            hidden_size=hidden_size,
            num_heads=num_heads,
            proj_type="perceptron",
            pos_embed_type="learnable",  # intra-organ (noop if patch_hw=224)
            dropout_rate=0.0,
            spatial_dims=2,
        )

        # 🔹 Organ-level positional embedding (THIS IS NEW)
        self.organ_pos_embed = nn.Parameter(
            torch.zeros(1, num_organs, hidden_size)
        )
        nn.init.trunc_normal_(self.organ_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(drop_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            batch_first=True,
            norm_first=True,
            dropout=drop_rate,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.norm = nn.LayerNorm(hidden_size)

        # Coordinate regression head
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, num_organs * 2)  # 16 landmarks * 2 coords
        )

    def forward(self, x):
        """
        Args:
            x: (B, 16, 224, 224)
            coords: (B, 16, 2)
        """

        B, T, H, W = x.shape
        assert T == self.num_organs, f"Expected {self.num_organs} organs, got {T}"

        x = x.view(B * T, 1, H, W)          # (B*16, 1, 224, 224)

        x = self.patch_embedding(x)         # (B*16, 1, hidden)
        x = x.mean(dim=1)                   # (B*16, hidden)
        x = x.view(B, T, -1)                # (B, 16, hidden)

        x = x + self.organ_pos_embed        # (B, 16, hidden)
        x = self.pos_drop(x)

        x = self.transformer(x)             # (B, 16, hidden)
        x = self.norm(x)

        coords = self.coord_head(x)          # (B, 16, 2)
        coords = coords.view(B, self.num_organs, 2)
        
        coords = torch.sigmoid(coords)      # normalize to [0, 1]

        return coords


class Swin_CR(nn.Module):
    def __init__(
        self,
        img_size=224,
        in_chans=1,
        num_landmarks=16,
        model_size="tiny",
        drop_rate=0.1,
    ):
        super().__init__()

        configs = {
            "tiny":  {"embed_dim": 96,  "depths": [2, 2, 6, 2],  "num_heads": [3, 6, 12, 24]},
            "small": {"embed_dim": 96,  "depths": [2, 2, 18, 2], "num_heads": [3, 6, 12, 24]},
            "base":  {"embed_dim": 128, "depths": [2, 2, 18, 2], "num_heads": [4, 8, 16, 32]},
        }

        cfg = configs[model_size]

        self.backbone = SwinTransformer(
            spatial_dims=2,              
            patch_size=4,
            in_chans=in_chans,
            embed_dim=cfg["embed_dim"],
            depths=cfg["depths"],
            num_heads=cfg["num_heads"],
            window_size=(7, 7),
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            drop_path_rate=0.2,
        )

        # Final feature dimension = embed_dim * 2^(num_stages - 1)
        self.feature_dim = 1536 
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.coord_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(drop_rate),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(drop_rate),

            nn.Linear(256, num_landmarks * 2),
        )

    def forward(self, x):
        """
        x: (B, 1, 224, 224)
        returns: (B, 16, 2) in [0, 1]
        """
        B = x.shape[0]

        feats = self.backbone(x)     # list of feature maps
        x = feats[-1]               # last stage (B, C, H, W)

        x = self.pool(x).flatten(1) # (B, C)
        coords = self.coord_head(x) # (B, 32)
        coords = torch.sigmoid(coords)

        return coords.view(B, -1, 2)
    
class ViT_Global_attn(nn.Module):
    """
    ViT with Global Attn
    Input: (B, 1, 224, 224)
    Outputs (B, 16, 2) - x,y coordinates (uses the cls token for coordinate regression)
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=1,
        embed_dim=256,
        depth=12,
        num_heads=8,
        num_landmarks=16,
        drop_rate=0.1
    ):
        super().__init__()

        assert img_size % patch_size == 0, "Image size must be divisible by patch size"

        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.num_landmarks = num_landmarks
        self.img_size = img_size

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.transformer = TransformerEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            drop_rate=drop_rate
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Coordinate regression head (CLS-based)
        self.coord_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, num_landmarks * 2)
        )

    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W) input CT slices
            coords: (B, num_landmarks, 2) normalized [0, 1]
        """
        B, _, H, W = x.shape

        x = self.patch_embed(x)  # (B, N, D)

        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls, x), dim=1)          # (B, N+1, D)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.transformer(x)  # (B, N+1, D)
        x = self.norm(x)

        cls_out = x[:, 0]  # (B, D)

        coords = self.coord_head(cls_out)  # (B, num_landmarks * 2)
        coords = coords.reshape(B, self.num_landmarks, 2)

        # Normalize to [0, 1]
        coords = torch.sigmoid(coords)

        return coords


if __name__ == "__main__":
    dummy_input = torch.randn(2, 1, 224, 224)  
    model = ViT_Global_attn(
        img_size=224, 
        patch_size=16, 
        in_channels=1, 
        embed_dim=256, 
        depth=12, 
        num_heads=8, 
        num_landmarks=16, 
        drop_rate=0.1)
    out = model(dummy_input)
    print("Model output shape:", out.shape)  # (2, 16, 2)

 
import torch

def mse_coordinate_loss(pred_coords, gt_coords, presence):
    """
    Args:
        pred_coords: (B, 16, 2) in [0, 1]
        gt_coords: (B, 16, 2) in [0, 1]
        presence: (B, 16)
    """
    mse = (pred_coords - gt_coords) ** 2  # (B, 16, 2)
    mse = mse.sum(dim=-1)  # (B, 16) - sum x² and y²
    
    mse = mse * presence
    
    return mse.sum() / (presence.sum() + 1e-6)

def CR_metrics_coordinates(pred_coords, gt_coords, presence, 
                                   segments_to_include, threshold=10.0, img_size=224):

    segment_mask = torch.zeros_like(presence)
    for seg_id in segments_to_include:
        segment_mask[:, seg_id] = 1.0
    
    effective_presence = presence * segment_mask

    pred_xy = pred_coords * img_size
    gt_xy = gt_coords * img_size
    
    diff = pred_xy - gt_xy
    edist = torch.sqrt((diff ** 2).sum(dim=-1))  # (B, 16)
    
    edist = edist * effective_presence
    
    metric1 = edist.sum() / (effective_presence.sum() + 1e-6)
    
    success = (edist < threshold).float() * effective_presence
    metric2 = success.sum() / (effective_presence.sum() + 1e-6)
    
    return metric1, metric2

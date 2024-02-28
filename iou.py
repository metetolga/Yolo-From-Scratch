import torch
from typing import List

def intersection_over_union(box_pred, box_true, box_format="midpoint"):
    # box_pred: (N, 4) where N is the number of bboxes
    # box_true: (N, 4)
    # Reason for slicing the array is keeping the shape same
    # indexing gives (N) shaped array 
    # slicing gives (N, 1) shaped array
    
    if box_format == "corners":
        box1_x1 = box_pred[..., 0:1] 
        box1_y1 = box_pred[..., 1:2]    
        box1_x2 = box_pred[..., 2:3]    
        box1_y2 = box_pred[..., 3:4]
        
        box2_x1 = box_true[..., 0:1] 
        box2_y1 = box_true[..., 1:2]    
        box2_x2 = box_true[..., 2:3]    
        box2_y2 = box_true[..., 3:4]     
    elif box_format == "midpoint":
        box1_x1 = box_pred[..., 0:1] - box_pred[..., 2:3] / 2
        box1_y1 = box_pred[..., 1:2] - box_pred[..., 3:4] / 2
        box1_x2 = box_pred[..., 0:1] + box_pred[..., 2:3] / 2
        box1_y2 = box_pred[..., 1:2] + box_pred[..., 3:4] / 2
        box2_x1 = box_true[..., 0:1] - box_true[..., 2:3] / 2
        box2_y1 = box_true[..., 1:2] - box_true[..., 3:4] / 2
        box2_x2 = box_true[..., 0:1] + box_true[..., 2:3] / 2
        box2_y2 = box_true[..., 1:2] + box_true[..., 3:4] / 2

    
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union  = box1_area + box2_area - intersection + 1e-6
    return intersection / union
    
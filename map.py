import torch
from collections import Counter
from iou import intersection_over_union

def mean_average_precision(pred_boxes, true_boxes, iou_th = 0, box_format="corners", num_classes = 20):
    # box_pred includes all prediction bounding boxes
    # bbox structure: [train_idx, class_pred, prob_score, x1, y1, x2, y2]
    # box_pred = [[bbox1], [bbox2], ...]
     
    average_pre = []
    ep = 1e-6
    
    for cls in range(num_classes):
        detections = []
        ground_truths = []
        
        for pred_box in pred_boxes:
            if pred_box[1] == cls:
                detections.append(pred_box)
                
        for true_box in true_boxes:
            if true_box[1] == cls:
                ground_truths.append(true_box)
    
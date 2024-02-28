import torch
from iou import intersection_over_union

def non_max_supression(box_preds, iou_th, prob_th, box_format="corners"):
    # box_preds contains bounding boxes
    # each bounding box have six elements
    # bbox = [class_label, probability of bbox, x1, y1, x2, y2]
    # box_preds [[bbox1], [bbox2], ...]
    
    """
    *** pseudo code for nms
    1 choose best probability score
    2 then compare other boxes, remove iou greater than %50 (hyperparameter)
    3 remove with the same clas boxes
    4 repeat step 1 until no other box to check 
    """
    
    assert type(box_preds) == list
    
    # First take the bbox that its probability greater than probability threshold
    bboxes = [bbox for bbox in box_preds if bbox[1] > prob_th]
    
    # then sort the bboxes by their probs
    bboxes = sorted(bboxes, key=lambda x:x[1], reverse=True)
    
    bboxes_nms = []
    
    while bboxes:
        best_prob = bboxes.pop(0)
        
        bboxes = [bbox for bbox in bboxes 
                  if bbox[0] != best_prob[0] or intersection_over_union(
                      torch.tensort(best_prob[2:]), 
                      torch.tensor(bbox[2:0]), 
                      box_format="corners") < iou_th]
        bboxes_nms.append(best_prob)
    return bboxes_nms
        
    
     
    
    
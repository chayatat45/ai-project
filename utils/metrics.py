import torch
from utils.box_utils import iou_of

def calculate_precision_recall(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_threshold=0.5):
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for i in range(len(pred_boxes)):
        preds = pred_boxes[i]
        gts = gt_boxes[i]

        matched = torch.zeros(gts.size(0)).to(preds.device)

        for p in preds:
            ious = iou_of(p.unsqueeze(0), gts)
            max_iou, idx = ious.max(0)

            if max_iou > iou_threshold and matched[idx] == 0:
                true_positive += 1
                matched[idx] = 1
            else:
                false_positive += 1

        false_negative += (matched == 0).sum().item()

    precision = true_positive / (true_positive + false_positive + 1e-6)
    recall = true_positive / (true_positive + false_negative + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return precision, recall, f1

def calculate_map(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_thresholds=[0.5]):
    aps = []
    for threshold in iou_thresholds:
        precision, recall, _ = calculate_precision_recall(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_threshold=threshold)
        aps.append(precision)

    return sum(aps) / len(aps)

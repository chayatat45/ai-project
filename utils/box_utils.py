import torch

def iou_of(boxes0, boxes1, eps=1e-5):
    overlap_top_left = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_bottom_right = torch.min(boxes0[..., 2:], boxes1[..., 2:])
    overlap_area = torch.prod(overlap_bottom_right - overlap_top_left, dim=-1).clamp(min=0)

    area0 = torch.prod(boxes0[..., 2:] - boxes0[..., :2], dim=-1)
    area1 = torch.prod(boxes1[..., 2:] - boxes1[..., :2], dim=-1)

    return overlap_area / (area0 + area1 - overlap_area + eps)

def match(threshold, truths, priors, labels, loc_t, conf_t, idx):
    overlaps = iou_of(truths.unsqueeze(1), priors.unsqueeze(0))
    best_prior_overlap, best_prior_idx = overlaps.max(1)
    best_truth_overlap, best_truth_idx = overlaps.max(0)

    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(0)

    best_truth_overlap.index_fill_(0, best_prior_idx, 2)

    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    matches = truths[best_truth_idx]
    conf = labels[best_truth_idx]
    conf[best_truth_overlap < threshold] = 0

    loc = encode(matches, priors)

    loc_t[idx] = loc
    conf_t[idx] = conf

def encode(matched, priors):
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - (priors[:, :2] + priors[:, 2:]) / 2
    g_wh = (matched[:, 2:] - matched[:, :2]) / (priors[:, 2:] - priors[:, :2])
    return torch.cat([g_cxcy, g_wh], 1)

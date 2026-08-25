from typing import List, NamedTuple, Optional, Sequence


class Box(NamedTuple):
    x1: float
    y1: float
    x2: float
    y2: float


def box_iou(a: Box, b: Box) -> float:
    inter_w = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    inter_h = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


class Match(NamedTuple):
    pred_index: Optional[int]
    gt_index: Optional[int]
    iou: Optional[float]
    status: str  # "tp" | "fp" | "fn"


def match_predictions_to_gt(
    pred_boxes: Sequence[Box],
    pred_confidences: Sequence[float],
    gt_boxes: Sequence[Box],
    iou_threshold: float,
) -> List[Match]:
    """Greedy, deterministic one-to-one matching. See module docstring for the exact rule."""
    pred_order = sorted(range(len(pred_boxes)), key=lambda i: (-pred_confidences[i], i))
    matched_gt = set()
    matches: List[Match] = []

    for pred_index in pred_order:
        best_gt_index, best_iou = None, 0.0
        for gt_index, gt_box in enumerate(gt_boxes):
            if gt_index in matched_gt:
                continue
            iou = box_iou(pred_boxes[pred_index], gt_box)
            if iou >= iou_threshold and (best_gt_index is None or iou > best_iou):
                best_gt_index, best_iou = gt_index, iou

        if best_gt_index is not None:
            matched_gt.add(best_gt_index)
            matches.append(Match(pred_index=pred_index, gt_index=best_gt_index, iou=best_iou, status="tp"))
        else:
            matches.append(Match(pred_index=pred_index, gt_index=None, iou=None, status="fp"))

    for gt_index in range(len(gt_boxes)):
        if gt_index not in matched_gt:
            matches.append(Match(pred_index=None, gt_index=gt_index, iou=None, status="fn"))

    return matches

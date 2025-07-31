from shapely.geometry import Polygon
import numpy as np

def polygon_iou(box1, box2):
    poly1 = Polygon(np.array(box1).reshape(4, 2))
    poly2 = Polygon(np.array(box2).reshape(4, 2))

    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return inter / union if union > 0 else 0.0

def rotated_nms(boxes, scores, iou_thresh=0.3):
    assert len(boxes) == len(scores)
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    keep = []
    while indices:
        current = indices.pop(0)
        keep.append(current)
        filtered_indices = []

        for i in indices:
            iou = polygon_iou(boxes[current], boxes[i])
            if iou < iou_thresh:
                filtered_indices.append(i)
        indices = filtered_indices

    kept_boxes = [boxes[i] for i in keep]
    return kept_boxes

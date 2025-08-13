import numpy as np
import torch

def calculate_iou(boxA, boxB):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    Args:
        boxA, boxB: [x1, y1, x2, y2] or [[x1, y1, x2, y2]]
    Returns:
        float: IoU value.
    """
    # Unpack if input is [[x1, y1, x2, y2]]
    if isinstance(boxA[0], (list, tuple, np.ndarray)):
        boxA = boxA[0]
    if isinstance(boxB[0], (list, tuple, np.ndarray)):
        boxB = boxB[0]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def calculate_mAP(predictions, ground_truths, iou_threshold=0.5):
    """
    Calculate mean Average Precision (mAP) for object detection.
    Args:
        predictions (list): List of predicted boxes per image.
        ground_truths (list): List of ground truth boxes per image.
        iou_threshold (float): IoU threshold for a positive match.
    Returns:
        float: mAP score.
    """
    # This is a simplified placeholder implementation
    # For each image, count matches above IoU threshold
    average_precisions = []
    for pred_boxes, gt_boxes in zip(predictions, ground_truths):
        matches = 0
        for pb in pred_boxes:
            for gb in gt_boxes:
                if calculate_iou(pb, gb) >= iou_threshold:
                    matches += 1
                    break
        precision = matches / (len(pred_boxes) + 1e-6)
        average_precisions.append(precision)
    mAP = np.mean(average_precisions)
    return mAP

def evaluate_model(model, data_loader, device):
    """
    Evaluate the model on the provided data loader.
    Args:
        model: PyTorch model.
        data_loader: DataLoader for evaluation.
        device: torch.device.
    Returns:
        float: mAP score.
    """
    model.eval()
    all_predictions = []
    all_ground_truths = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            outputs = model(images)
            # Assume outputs and targets are lists of boxes
            all_predictions.append(outputs)
            all_ground_truths.append(targets)
    return calculate_mAP(all_predictions, all_ground_truths)
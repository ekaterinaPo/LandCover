import numpy as np

def calculate_iou(y_true, y_pred, num_classes):
    ious = []
    y_true_classes = np.argmax(y_true.detach().numpy(), axis = 1)
    y_pred_classes = np.argmax(y_pred.detach().numpy(), axis = 1)
    
    for class_id in range(num_classes):
        true_class_mask = (y_true_classes == class_id).astype(int)
        pred_class_mask = (y_pred_classes == class_id).astype(int)

        intersection = np.logical_and(true_class_mask, pred_class_mask).sum()
        union = np.logical_or(true_class_mask, pred_class_mask).sum()

        iou = intersection / (union + 1e-6)
        ious.append(iou)
    
    return ious

# def accuracy(outputs, labels, num_classes):
def accuracy(y_true, y_pred, num_classes):
    ious = calculate_iou(y_true, y_pred, num_classes)
    return sum(ious) / len(ious)
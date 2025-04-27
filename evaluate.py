import torch
from models.ssd import SSD
from datasets.acne_dataset import AcneDataset
from utils.metrics import calculate_precision_recall, calculate_map
from torch.utils.data import DataLoader
from configs.config import Config

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AcneDataset(Config.val_images, Config.val_annotations, transform=False)
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4)

    model = SSD(num_classes=Config.num_classes)
    model.load_state_dict(torch.load(Config.model_path))
    model.to(device)
    model.eval()

    pred_boxes = []
    pred_labels = []
    pred_scores = []
    gt_boxes = []
    gt_labels = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)

            # TODO: decode outputs to boxes, labels, scores (placeholder)

    precision, recall, f1 = calculate_precision_recall(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels)
    map_50 = calculate_map(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_thresholds=[0.5])
    map_50_95 = calculate_map(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_thresholds=[x/100 for x in range(50, 100, 5)])

    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, mAP@0.5: {map_50:.4f}, mAP@0.5-0.95: {map_50_95:.4f}")

if __name__ == "__main__":
    evaluate()

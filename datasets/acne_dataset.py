import torch
from torch.utils.data import Dataset
import os
import glob
import cv2
import json

class ACNEDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        super(ACNEDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.split = split  # 'train', 'val', 'test'

        # รูปภาพ
        self.images = sorted(glob.glob(os.path.join(root, split, 'images', '*.jpg')))

        # Labels (สมมุติใช้เป็น JSON files)
        self.labels = sorted(glob.glob(os.path.join(root, split, 'labels', '*.json')))

        assert len(self.images) == len(self.labels), "Mismatch between images and labels!"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Load image
        img_path = self.images[index]
        label_path = self.labels[index]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load bounding boxes
        with open(label_path, 'r') as f:
            boxes = json.load(f)

        boxes = torch.tensor(boxes['boxes'], dtype=torch.float32)  # [N, 4]
        labels = torch.ones((boxes.size(0),), dtype=torch.long)    # All labels are 1 ("สิว")

        target = {'boxes': boxes, 'labels': labels}

        if self.transform:
            img, target = self.transform(img, target)

        return img, target

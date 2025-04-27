import cv2
import numpy as np
import random
import torch

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target

class ToTensor(object):
    def __call__(self, img, target):
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img, target

class Resize(object):
    def __init__(self, size=512):
        self.size = size

    def __call__(self, img, target):
        h, w, _ = img.shape
        img = cv2.resize(img, (self.size, self.size))

        if 'boxes' in target:
            boxes = target['boxes']
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * self.size / w
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * self.size / h
            target['boxes'] = boxes

        return img, target

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = cv2.flip(img, 1)
            if 'boxes' in target:
                boxes = target['boxes']
                boxes[:, [0, 2]] = img.shape[1] - boxes[:, [2, 0]]
                target['boxes'] = boxes
        return img, target

class Normalize(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img, target):
        img = (img - torch.tensor(self.mean).view(3, 1, 1)) / torch.tensor(self.std).view(3, 1, 1)
        return img, target

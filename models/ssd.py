import torch
import torch.nn as nn
from models.efficientnetv2_cbam import EfficientNetV2_CBAM

class SSD(nn.Module):
    def __init__(self, num_classes):
        super(SSD, self).__init__()

        self.backbone = EfficientNetV2_CBAM(pretrained=True)

        # เพิ่ม Conv layers ทำ feature maps ต่อ
        self.extras = nn.Sequential(
            nn.Conv2d(160, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # จำนวน prior boxes ที่แต่ละ feature map location
        self.num_defaults = 4

        # Localization heads (predict bounding box offsets)
        self.loc = nn.ModuleList([
            nn.Conv2d(160, self.num_defaults * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, self.num_defaults * 4, kernel_size=3, padding=1),
            nn.Conv2d(512, self.num_defaults * 4, kernel_size=3, padding=1)
        ])

        # Confidence heads (predict class probabilities)
        self.conf = nn.ModuleList([
            nn.Conv2d(160, self.num_defaults * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, self.num_defaults * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(512, self.num_defaults * num_classes, kernel_size=3, padding=1)
        ])

    def forward(self, x):
        sources = []

        # Extract feature maps
        x = self.backbone(x)
        sources.append(x)

        # ต่อด้วย extras
        for layer in self.extras:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                sources.append(x)

        # Apply multibox head to sources
        locs = []
        confs = []

        for (x, l, c) in zip(sources, self.loc, self.conf):
            locs.append(l(x).permute(0, 2, 3, 1).contiguous())
            confs.append(c(x).permute(0, 2, 3, 1).contiguous())

        locs = torch.cat([o.view(o.size(0), -1) for o in locs], 1)
        confs = torch.cat([o.view(o.size(0), -1) for o in confs], 1)

        return locs.view(locs.size(0), -1, 4), confs.view(confs.size(0), -1, self.num_defaults)


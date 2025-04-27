import torch
import torch.nn as nn
import torchvision.models as models
from utils.attention_modules import CBAM

class EfficientNetV2_CBAM(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientNetV2_CBAM, self).__init__()
        model = models.efficientnet_v2_l(pretrained=pretrained)  # ใช้ EfficientNetV2-L แทน B8 (เพราะใน torchvision ไม่มี B8, แต่ L = ใหญ่ใกล้เคียง)

        self.stem = model.features[0]
        self.blocks = nn.ModuleList()
        for block in model.features[1:]:
            self.blocks.append(nn.Sequential(
                block,
                CBAM(block[0].out_channels)
            ))
        
    def forward(self, x):
        x = self.stem(x)
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features  # ส่งออกหลาย scale ให้ SSD

def efficientnetv2_cbam(pretrained=True):
    return EfficientNetV2_CBAM(pretrained=pretrained)

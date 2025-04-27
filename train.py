import torch
from torch.utils.data import DataLoader
from utils.dataloader import AcneDataset, get_train_transforms
from models.ssd_with_backbone import SSD
import torch.optim as optim
import torch.nn.functional as F
from config import *

def train():
    dataset = AcneDataset(
        images_dir=f"{DATASET_PATH}/images",
        labels_dir=f"{DATASET_PATH}/labels",
        transform=get_train_transforms()
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    model = SSD(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for images, targets in dataloader:
            images = torch.stack(images).to(DEVICE)
            optimizer.zero_grad()

            cls_preds, bbox_preds = model(images)

            # (ทำ Loss แบบง่าย)
            loss_cls = F.cross_entropy(cls_preds.view(-1, NUM_CLASSES), torch.cat([t["labels"] for t in targets]).to(DEVICE))
            loss_bbox = F.l1_loss(bbox_preds.view(-1, 4), torch.cat([t["boxes"] for t in targets]).to(DEVICE))

            loss = loss_cls + loss_bbox
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(dataloader):.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), SAVE_MODEL_PATH)

if __name__ == "__main__":
    train()

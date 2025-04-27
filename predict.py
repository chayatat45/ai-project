import torch
from models.ssd import SSD
from configs.config import Config
from PIL import Image
import torchvision.transforms as T

def predict(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SSD(num_classes=Config.num_classes)
    model.load_state_dict(torch.load(Config.model_path))
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((Config.input_size, Config.input_size)),
        T.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

    # TODO: decode outputs to boxes, labels, scores (placeholder)

    return outputs

if __name__ == "__main__":
    predict('path/to/your/image.jpg')

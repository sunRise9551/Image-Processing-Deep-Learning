import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
from model import AlexNet

data_transforms = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

image = Image.open("../data_set/tulip.png")
image = data_transforms(image)
image = torch.unsqueeze(image, dim=0)   # expand batch dimension

try:
    json_file = open("./class_indices.json", 'r')
    class_dict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

model = AlexNet(num_classes=5)
model_weight_path = "./AlexNet.pth"
model.load_state_dict(torch.load(model_weight_path))
model.eval()

with torch.no_grad():
    output = torch.squeeze(model(image))    # Squeeze Batch dimension
    predict = torch.softmax(output, dim=0)
    predict_class = torch.argmax(predict).numpy()

print(class_dict[str(predict_class)], predict[predict_class].item())
plt.show()



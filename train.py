import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
import VGG19
import ResNet50
import matplotlib.pyplot as plt

class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.chosen_features = ["0", "5", "10", "19", "28"]
        self.model = VGG19.VGG19()

    def forward(self, x):
        features = []

        for layer_num, layer in enumerate(self.model.net):
            x = layer(x)

            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ResNet50.ResNet50()

    def forward(self, x):
        features = []
        for stage in [self.model.stage1, self.model.stage2, self.model.stage3]:
            for layer in stage:
                x = layer(x)
            features.append(x)
        return features



def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 356

loader = transforms.Compose(
    [
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor(),
    ]
)

original_img = load_image("06.jpg")
style_img = load_image("style.jpg")


generated = original_img.clone().requires_grad_(True)
model = ResNet().to(device).eval()

total_steps = 60000
learning_rate = 0.001
alpha = 1
beta = 0.01
optimizer = optim.Adam([generated], lr=learning_rate)

for step in range(total_steps):
    # Obtain the convolution features in specifically chosen layers
    generated_features = model(generated)
    original_img_features = model(original_img)
    style_features = model(style_img)

    # Loss is 0 initially
    style_loss = original_loss = 0

    # iterate through all the features for the chosen layers
    for gen_feature, orig_feature, style_feature in zip(
        generated_features, original_img_features, style_features
    ):

        # batch_size will just be 1
        batch_size, channel, height, width = gen_feature.shape
        original_loss += torch.mean((gen_feature - orig_feature) ** 2)
        # Compute Gram Matrix of generated
        G = gen_feature.view(channel, height * width).mm(
            gen_feature.view(channel, height * width).t()
        )
        # Compute Gram Matrix of Style
        A = style_feature.view(channel, height * width).mm(
            style_feature.view(channel, height * width).t()
        )
        style_loss += torch.mean((G - A) ** 2)

    total_loss = alpha * original_loss + beta * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 1000 == 0:
        print(total_loss)
        save_image(generated, "ResNet50-generated.png")

torch.save(model, './ResNet50.pt')
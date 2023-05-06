#import libraries

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import os
from PIL import Image
import torch.utils.data as data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training on {device}')

#define hyperparameters

image_size = 256
input_nc = 3
output_nc = 3
lr = 0.0002
batch_size = 1
num_epochs = 2
lambda_cyc = 10
#Test
 # Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(in_channels),
        )

    def forward(self, x):
        return x + self.conv_block(x)

# Generator
class Generator(nn.Module):
    def __init__(self, input_channels, output_channels, num_res_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        in_channels = 64
        out_channels = in_channels * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels
            out_channels = in_channels * 2

        # Residual blocks
        for _ in range(num_res_blocks):
            model += [ResidualBlock(in_channels)]

        # Upsampling
        out_channels = in_channels // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels
            out_channels = in_channels // 2

        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_channels, kernel_size=7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()

        def discriminator_block(in_channels, out_channels, normalization=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, kernel_size=4, padding=1)
        )

    def forward(self, x):
        return self.model(x)

# Load the saved weights
G_A2B = Generator(input_nc, output_nc)

G_B2A = Generator(output_nc, input_nc)

G_A2B.load_state_dict(torch.load('G.pth'))
G_A2B = G_A2B.to(device)
G_B2A.load_state_dict(torch.load('F.pth'))
G_B2A = G_B2A.to(device)
# Set the models to evaluation mode
G_A2B.eval()
G_B2A.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Load the test images
image_A = Image.open("hz/testA/1.jpg").convert("RGB")
image_B = Image.open("hz/testB/1.jpg").convert("RGB")
# Apply the transformations
image_A = transform(image_A)
image_B = transform(image_B)
# Add a batch dimension and move the data to the device
image_A = image_A.unsqueeze(0).to(device)
image_B = image_B.unsqueeze(0).to(device)
# Generate the output images
with torch.no_grad():
    fake_B = G_A2B(image_A)
    fake_A = G_B2A(image_B)

# Post-process the generated images
def tensor_to_image(tensor):
    tensor = tensor.cpu()
    image = 0.5 * (tensor.detach().numpy() + 1)
    image = image.clip(0, 1)
    image = (image * 255).astype("uint8")
    return image.transpose(1, 2, 0)

generated_image_A = tensor_to_image(fake_A.squeeze())
generated_image_B = tensor_to_image(fake_B.squeeze())

# Save the generated images
Image.fromarray(generated_image_A).save("generated_image_A.png")
Image.fromarray(generated_image_B).save("generated_image_B.png")
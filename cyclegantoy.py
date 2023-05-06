
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
num_epochs = 200
lambda_cyc = 10
save_interval = 1

class CycleGANDataset(data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = sorted(os.listdir(root_dir))
        self.transform = transform

    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir, self.image_paths[index])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_paths)


""" class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_dropout=False, activation=nn.ReLU(inplace=True)):
        super(UNetBlock, self).__init__()
        self.down = down
        self.activation = activation

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1) if down else nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.dropout = nn.Dropout(0.5) if use_dropout else None

    def forward(self, x):
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        if self.dropout:
            x = self.dropout(x)
        return x

class Generator(nn.Module):
    def __init__(self, input_channels, output_channels, num_filters=64):
        super(Generator, self).__init__()

        # Encoder (downsampling)
        self.enc1 = UNetBlock(input_channels, num_filters, down=True, activation=nn.LeakyReLU(0.2, inplace=True))
        self.enc2 = UNetBlock(num_filters, num_filters * 2, down=True)
        self.enc3 = UNetBlock(num_filters * 2, num_filters * 4, down=True)
        self.enc4 = UNetBlock(num_filters * 4, num_filters * 8, down=True)

        # Decoder (upsampling)
        self.dec1 = UNetBlock(num_filters * 8, num_filters * 4, down=False, use_dropout=True)
        self.dec2 = UNetBlock(num_filters * 4 * 2, num_filters * 2, down=False, use_dropout=True)
        self.dec3 = UNetBlock(num_filters * 2 * 2, num_filters, down=False, use_dropout=True)
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(num_filters * 2, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        dec1 = self.dec1(enc4)
        dec1 = torch.cat((dec1, enc3), 1)
        dec2 = self.dec2(dec1)
        dec2 = torch.cat((dec2, enc2), 1)
        dec3 = self.dec3(dec2)
        dec3 = torch.cat((dec3, enc1), 1)

        output = self.dec4(dec3)
        return output
    
class Discriminator(nn.Module):
    def __init__(self, input_channels, num_filters=64):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            # Layer 1
            nn.Conv2d(input_channels, num_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2
            nn.Conv2d(num_filters, num_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(num_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3
            nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(num_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4
            nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Output layer
            nn.Conv2d(num_filters * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.layers(x)
 """

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


#init models

G = Generator(input_nc, output_nc)
G = G.to(device)
F = Generator(output_nc, input_nc)
F = F.to(device)
D_X = Discriminator(input_nc)
D_X = D_X.to(device)
D_Y = Discriminator(output_nc)
D_Y = D_Y.to(device)

#set up data
transform = transforms.Compose([transforms.Resize((256, 256)),  # Resize the image to 256x256 pixels
    transforms.ToTensor(),  # Convert the PIL image to a PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image tensor)
])
dataset_X = CycleGANDataset("hz/trainA", transform=transform)
dataset_Y = CycleGANDataset("hz/trainB", transform=transform)
dataloader_X = DataLoader(dataset_X, batch_size=batch_size, shuffle=True, num_workers=2)
dataloader_Y = DataLoader(dataset_Y, batch_size=batch_size, shuffle=True, num_workers=2)


#set up optimizers
optimizer_G = optim.Adam(G.parameters(), lr=lr)
optimizer_F = optim.Adam(F.parameters(), lr=lr)
optimizer_D_X = optim.Adam(D_X.parameters(), lr=lr)
optimizer_D_Y = optim.Adam(D_Y.parameters(), lr=lr)

#set up loss functions
adversarial_loss = nn.MSELoss()
cycle_consistency_loss = nn.L1Loss()

#training loop
for epoch in range(num_epochs):
    print(epoch)
    # Training steps:
    # 1. Update generators G and F
    # 2. Update discriminators D_X and D_Y
    # 3. Calculate and log losses
    for batch_X, batch_Y in zip(dataloader_X, dataloader_Y):
        batch_X  = [b.to(device) for b in batch_X]
        batch_Y = [c.to(device) for c in batch_Y]
        real_X = batch_X[0]
        real_Y = batch_Y[0]

        # Train generators
        optimizer_G.zero_grad()

        fake_Y = G(real_X)
        fake_X = F(real_Y)

        D_Y_pred = D_Y(fake_Y)
        D_X_pred = D_X(fake_X)

        loss_G = adversarial_loss(D_Y_pred, torch.ones_like(D_Y_pred)) + adversarial_loss(D_X_pred, torch.ones_like(D_X_pred))

        cyc_X = F(fake_Y)
        cyc_Y = G(fake_X)
        
        loss_cycle = cycle_consistency_loss(cyc_X, real_X) + cycle_consistency_loss(cyc_Y, real_Y)

        loss_G_total = loss_G + lambda_cyc * loss_cycle
        loss_G_total.backward()
        optimizer_G.step()

        # Train discriminators
        optimizer_D_X.zero_grad()
        optimizer_D_Y.zero_grad()

        D_X_real = D_X(real_X)
        D_Y_real = D_Y(real_Y)
        D_X_fake = D_X(fake_X.detach())
        D_Y_fake = D_Y(fake_Y.detach())

        loss_D_X = adversarial_loss(D_X_real, torch.ones_like(D_X_real)) + adversarial_loss(D_X_fake, torch.zeros_like(D_X_fake))
        loss_D_Y = adversarial_loss(D_Y_real, torch.ones_like(D_Y_real)) + adversarial_loss(D_Y_fake, torch.zeros_like(D_Y_fake))

        loss_D_X.backward()
        loss_D_Y.backward()
        optimizer_D_X.step()
        optimizer_D_Y.step()
        print(loss_G_total)
    # Save the model weights every `save_interval` epochs
    if (epoch + 1) % save_interval == 0:
        torch.save(G.state_dict(), f'G_{epoch + 1}.pth')
        torch.save(F.state_dict(), f'F_{epoch + 1}.pth')
        torch.save(D_X.state_dict(), f'D_X_epoch_{epoch + 1}.pth')
        torch.save(D_Y.state_dict(), f'D_Y_epoch_{epoch + 1}.pth')


#save data
torch.save(G.state_dict(), "G.pth")
torch.save(F.state_dict(), "F.pth")
torch.save(D_X.state_dict(), "D_X.pth")
torch.save(D_Y.state_dict(), "G_X.pth")

#Test
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
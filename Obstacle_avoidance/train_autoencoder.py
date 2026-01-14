import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
input_size = 224 * 224 * 3
batch_size = 16
num_epochs = 20
learning_rate = 1e-3

noise_factor = 0.1
alpha = 0.8  # weight for MSE
beta = 0.2   # weight for SSIM


# Autoencoder Definition
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # [3, 224, 224] → [16, 112, 112]
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # [32, 56, 56]
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # [64, 28, 28]
            nn.ReLU(True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # output between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
# adding noise to input images
def add_noise(x, noise_factor=0.1):
    noisy = x + noise_factor * torch.randn_like(x)
    return torch.clamp(noisy, 0.0, 1.0)

# SSIM Loss Function
def ssim_loss(x, y, C1=0.01**2, C2=0.03**2):
    # x,y: (B,C,H,W) in [0,1]
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)

    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y

    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2)
    ssim = ssim_n / (ssim_d + 1e-8)
    return 1 - ssim.mean()  # minimize 1-SSIM



# Data Preparation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Only NORMAL (no obstacle) images
path_to_normal_data = 'beds_without_obstacles'
dataset = datasets.ImageFolder(path_to_normal_data, transform=transform)
loader = DataLoader(dataset, batch_size, shuffle=True)


# Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvAutoencoder().to(device)
mse = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), learning_rate)

for epoch in range(num_epochs):
    loss_sum = 0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        noisy_imgs = add_noise(imgs, noise_factor)
        outputs = model(noisy_imgs)
        loss = alpha * mse(outputs, imgs) + beta * ssim_loss(outputs, imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss_sum/len(loader):.6f}")

torch.save(model.state_dict(), 'autoencoder_obstacle.pth')
print("Autoencoder trained and saved.")


# Determine threshold from training data
errors = []
model.eval()
with torch.no_grad():
    for imgs, _ in loader:
        imgs = imgs.to(device)
        recon = model(imgs)
        err = torch.mean((imgs - recon) ** 2, dim=(1,2,3)).cpu().numpy()
        errors.extend(err)

threshold = np.percentile(errors, 99)  # 99th percentile
print("Threshold:", threshold)


# Inference (Detect Anomalies)
def anomaly_score(img):
    with torch.no_grad():
        img = img.unsqueeze(0).to(device)
        recon = model(img)
        error = torch.mean((img - recon) ** 2).item()
    return error

# If error > threshold → obstacle detected

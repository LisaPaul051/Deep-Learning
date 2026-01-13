import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Hyperparameters
input_size = 224 * 224 * 3
batch_size = 16
num_epochs = 20
learning_rate = 1e-3

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
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), learning_rate)

for epoch in range(num_epochs):
    loss_sum = 0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss_sum/len(loader):.6f}")

torch.save(model.state_dict(), 'autoencoder_obstacle.pth')
print("Autoencoder trained and saved.")


# Inference (Detect Anomalies)

def anomaly_score(img):
    with torch.no_grad():
        img = img.unsqueeze(0).to(device)
        recon = model(img)
        error = torch.mean((img - recon) ** 2).item()
    return error

# Threshold
threshold = 0.01
# If error > threshold → obstacle detected

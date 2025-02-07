import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import pandas as pd
import os

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define a custom dataset class for our Futhark images
class FutharkDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.rune_encoder = LabelEncoder()
        self.data['rune_encoded'] = self.rune_encoder.fit_transform(self.data['rune'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['file_path']
        img = Image.open(img_path).convert('RGB')  # Convert to RGB
        if self.transform:
            img = self.transform(img)
        label = self.data.iloc[idx]['rune_encoded']
        return img, label

# Set up the dataset and data loader
csv_file = 'runes_dataset.csv'
transform = transforms.Compose([transforms.ToTensor()])
dataset = FutharkDataset(csv_file, transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Print the first file in the dataset
for images, labels in data_loader:
    print(images.shape)
    break

# Define a simple CNN model
class FutharkModel(nn.Module):
    def __init__(self):
        super(FutharkModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3)
        self.fc1 = nn.Linear(16 * 10 * 10, 128)  # Adjusted to match 48x48 image size
        self.fc2 = nn.Linear(128, 24)  # Assuming 24 classes for the runes

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 10 * 10)  # Corrected size for 48x48 images after convolutions and poolings
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
model = FutharkModel()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(10):  # adjust the number of epochs based on your needs
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
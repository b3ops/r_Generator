import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
from sklearn.preprocessing import LabelEncoder
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import pandas as pd
import os
import copy

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Your FutharkDataset class definition here
class FutharkDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.rune_encoder = LabelEncoder()
        self.data['rune_encoded'] = self.rune_encoder.fit_transform(self.data['rune'])
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomCrop(48, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

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
dataset = FutharkDataset(csv_file)
#transform = transforms.Compose([transforms.ToTensor()])

# Assuming you have around 1000 images, adjust the split accordingly
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Print the first file in the dataset
for images, labels in val_loader:
    print(images.shape)
    break

# Your FutharkModel class definition here
class FutharkModel(nn.Module):
    def __init__(self):
        super(FutharkModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 24)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool(nn.functional.relu(self.bn2(self.conv2(x))))
        x = self.pool(nn.functional.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 6 * 6)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout1(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
model = FutharkModel()
model.to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(val_loader)

best_val_loss = float('inf')
patience = 50
counter = 0
best_model_wts = copy.deepcopy(model.state_dict())

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=500, patience=50):
    best_val_loss = float('inf')
    counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = validate(model, val_loader, criterion, device)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
            torch.save(model.state_dict(), 'futhark_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                model.load_state_dict(best_model_wts)
                break
        
        scheduler.step()
        print(f'Epoch {epoch+1}, Train Loss: {loss.item()}, Validation Loss: {val_loss}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def apply_wood_style(image, carv_folder, style_transfer_model):
    # Convert image to RGB for consistency
    image = image.convert('RGB')
    
    # Transform for both content and style image
    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to match content image size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    content_tensor = img_transform(image).unsqueeze(0).to(device)

    # Load and resize style image
    carving_images = [f for f in os.listdir(carv_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not carving_images:
        raise FileNotFoundError(f"No carving images found in {carv_folder}")
    style_image_path = os.path.join(carv_folder, carving_images[0])  # Using the first image for simplicity
    style_image = Image.open(style_image_path).convert('RGB')
    style_tensor = img_transform(style_image).unsqueeze(0).to(device)

    # Rest of your style transfer simulation
    with torch.no_grad():
        content_features = style_transfer_model(content_tensor)
        style_features = style_transfer_model(style_tensor)

        # Very simplistic 'style transfer'
        styled_features = content_features + 0.1 * style_features

        # Convert back to image (placeholder for real style transfer)
        styled_image = styled_features.cpu().squeeze(0).permute(1, 2, 0)[:3]
        styled_image = styled_image.clamp(0, 1)
        styled_image = transforms.ToPILImage()(styled_image)
    
    return styled_image.resize((48, 48), Image.LANCZOS)

def main():
    # Set up the dataset and data loader
    csv_file = 'runes_dataset.csv'
    dataset = FutharkDataset(csv_file)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize the model, loss function, and optimizer inside main()
    model = FutharkModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Train the model
    model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)
    
    print("Model training completed. Now entering rune translation mode.")
    
    # Create rune to translation mapping
    rune_to_translation = dataset.data.set_index('rune')['translation'].to_dict()
    translation_to_rune = {v.lower(): k for k, v in rune_to_translation.items()}

    # Style transfer model (placeholder)
    style_transfer_model = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()

    # Path to the folder containing wood carving images
    carv_folder = 'carv'

    print("Enter text to translate into runes (whole word):")
    while True:
        word = input("Enter a word (or 'q' to quit): ").strip().lower()
        if word == 'q':
            break

        output_data = []
        images = []
        
        for char in word:
            rune_name = translation_to_rune.get(char, None)
            if rune_name:
                matching_rows = dataset.data[dataset.data['rune'] == rune_name]
                if not matching_rows.empty:
                    img_path = matching_rows.iloc[0]['file_path']
                    image = Image.open(img_path).convert('RGB')
                    
                    # Apply wood carving style, passing carv_folder
                    styled_image = apply_wood_style(image, carv_folder, style_transfer_model)
                    save_path = f'generated_rune_{char}_styled.png'
                    styled_image.save(save_path)
                    images.append(styled_image)
                    
                    output_data.append({
                        'character': char,
                        'rune': rune_name,
                        'image_path': os.path.abspath(save_path)
                    })
                else:
                    print(f"No image found for rune '{rune_name}' for character '{char}'.")
            else:
                print(f"No rune found for '{char}'.")

        if output_data:
            # Combine images
            width = len(images) * 48
            height = 48
            combined_image = Image.new('RGB', (width, height))
            
            for i, img in enumerate(images):
                combined_image.paste(img, (i * 48, 0))
            
            combined_save_path = f'combined_runes_{word}_styled.png'
            combined_image.save(combined_save_path)

            # Create a DataFrame with the mappings
            output_df = pd.DataFrame(output_data)
            output_csv_path = f'output_runes_{word}_styled.csv'
            output_df.to_csv(output_csv_path, index=False)
            
            print(f"Generated and combined styled runes for '{word}' into {combined_save_path}")
            print(f"Mappings saved to {output_csv_path}")
            for entry in output_data:
                print(f"Character '{entry['character']}' -> Rune '{entry['rune']}' styled and saved as '{entry['image_path']}")
if __name__ == "__main__":
    main()
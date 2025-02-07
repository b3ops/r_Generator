import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import *
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from PIL import Image, ImageDraw
import os

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset class for runes
class ElderFutharkDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.style_encoder = LabelEncoder()
        self.rune_encoder = LabelEncoder()
        self.data['style_encoded'] = self.style_encoder.fit_transform(self.data['style'])
        self.data['rune_encoded'] = self.rune_encoder.fit_transform(self.data['rune'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['file_path']
        rune = self.data.iloc[idx]['rune']
        translation = self.data.iloc[idx]['translation']
        return img_path, rune, translation

# Transformations for runes (not used here but kept for consistency)
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset
dataset = ElderFutharkDataset('runes_dataset.csv', transform=None)

# Create rune to translation mapping
rune_to_translation = dataset.data.set_index('rune')['translation'].to_dict()

# Create translation to rune mapping for quick lookup
translation_to_rune = {v.lower(): k for k, v in rune_to_translation.items()}

# Load a pre-trained model for style transfer (placeholder)
style_transfer_model = models.vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()

# Path to the folder containing wood carving images (assumed to be in the same directory as script)
carv_folder = 'carv'

def apply_wood_style(image):
    # Convert image to RGB for consistency
    image = image.convert('RGB')
    
    # Transform for the model
    img_transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Use these values if you're using a model like VGG19
    ])
    img_tensor = img_transform(image).unsqueeze(0).to(device)

    # Here, we'll just use one random carving image for style transfer (for simplicity)
    carving_images = [f for f in os.listdir(carv_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    style_image_path = os.path.join(carv_folder, carving_images[0])  # Using the first image for simplicity
    style_image = Image.open(style_image_path).convert('RGB')
    style_tensor = img_transform(style_image).unsqueeze(0).to(device)

    # Placeholder for style transfer - we'll simulate a style transfer by mixing content and style
    with torch.no_grad():
        # Extract features from content and style images
        content_features = style_transfer_model(img_tensor)
        style_features = style_transfer_model(style_tensor)

        # Very simplistic 'style transfer' - just adding the features
        styled_features = content_features + 0.1 * style_features  # This is not real style transfer, just an example

        # Here, we would typically reconstruct the image from these features, but for simplicity:
        styled_image = styled_features  # This is incorrect for real style transfer, but works for this example

    # Convert back to PIL Image - this part needs careful handling
    styled_image = styled_image.cpu().squeeze(0).permute(1, 2, 0)[:3]  # Take only first 3 channels if more exist
    styled_image = styled_image.clamp(0, 1)  # Clamp to ensure values are between 0 and 1
    styled_image = transforms.ToPILImage()(styled_image)
    
    # Convert back to grayscale if the original was grayscale
    if image.mode == 'L':
        styled_image = styled_image.convert('L')
    
    # Resize back to original dimensions
    return styled_image.resize((48, 48), Image.LANCZOS)

def main():
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
                    image = Image.open(img_path).convert('L')  # Convert to grayscale if needed
                    
                    # Apply wood carving style
                    styled_image = apply_wood_style(image)
                    save_path = f'generated_rune_{char}_styled.png'
                    styled_image.save(save_path)
                    images.append(styled_image)
                    
                    # Record the mapping
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
            combined_image = Image.new('L', (width, height))
            
            for i, img in enumerate(images):
                combined_image.paste(img, (i * 48, 0))
            
            combined_save_path = f'combined_runes_{word}_styled.png'
            combined_image.save(combined_save_path)

            # Create a DataFrame with the mappings
            output_df = pd.DataFrame(output_data)
            # Save the DataFrame to a new CSV file
            output_csv_path = f'output_runes_{word}_styled.csv'
            output_df.to_csv(output_csv_path, index=False)
            
            print(f"Generated and combined styled runes for '{word}' into {combined_save_path}")
            print(f"Mappings saved to {output_csv_path}")
            for entry in output_data:
                print(f"Character '{entry['character']}' -> Rune '{entry['rune']}' styled and saved as '{entry['image_path']}'")

if __name__ == "__main__":
    main()
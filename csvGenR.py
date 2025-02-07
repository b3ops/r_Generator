import os
from PIL import Image

# Directory where your rune images are saved
rune_dir = 'img/'

# Dictionary for special cases
special_cases = {
    'eihwaz': 'ei',
    'ingwaz': 'ng'
}

# List to hold the rows of your CSV
csv_rows = []

# Iterate over all files in the directory
for filename in os.listdir(rune_dir):
    if filename.endswith('.png'):
        # Split the filename to get rune name and translation
        base_name = os.path.splitext(filename)[0]
        parts = base_name.split('-')
        
        if len(parts) == 2:
            rune_name, translation = parts
            # Check if it's a special case
            if rune_name in special_cases:
                translation = special_cases[rune_name]
        else:
            # If there's no '-' in the filename, we'll log an error and skip this file
            print(f"Error: File {filename} does not follow the expected naming convention 'rune-translation.png'. Skipping.")
            continue
        
        # Here you decide the style, let's say 'old_carved' for simplicity
        style = 'old_carved'
        
        csv_rows.append({
            'file_path': os.path.join(rune_dir, filename),
            'rune': rune_name,
            'translation': translation,
            'style': style
        })

# Write to CSV
import pandas as pd
df = pd.DataFrame(csv_rows)
df.to_csv('runes_dataset.csv', index=False)
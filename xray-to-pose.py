import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import json

# Define a color map for each label
color_map = {
    'left hip socket': (255, 0, 0),  # Red
    'right hip socket': (0, 255, 0),  # Green
    'left knee': (0, 0, 255),  # Blue
    'right knee': (255, 255, 0),  # Yellow
    'hip': (255, 165, 0),  # Orange
    'spine': (255, 0, 255)   # Magenta
}

# Load the CSV file without headers
csv_file = './leg-hip-annotations/point_annotations.csv'
df = pd.read_csv(csv_file, header=None)

# Manually set the column names
df.columns = ['label', 'x', 'y', 'file', 'width', 'height']

def draw_stick_figure(image, df):
    # Create a black image with the same dimensions as the original image
    img = np.zeros_like(image)
    height, width, _ = image.shape

    # Normalize line weight and point size
    line_weight = int(min(height, width) * 0.02)  # Adjust this factor as needed
    point_size = int(min(height, width) * 0.03)  # Adjust this factor as needed

    # Create a dictionary to store coordinates by label
    coords = {}
    for index, row in df.iterrows():
        label = row['label']
        x = int(row['x'])
        y = int(row['y'])
        coords[label] = (x, y)

    # Draw the connections based on the specified rules
    if 'left hip socket' in coords and 'left knee' in coords:
        cv2.line(img, coords['left hip socket'], coords['left knee'], (255, 255, 255), line_weight)
    if 'left hip socket' in coords and 'hip' in coords:
        cv2.line(img, coords['left hip socket'], coords['hip'], (255, 255, 255), line_weight)
    if 'right hip socket' in coords and 'right knee' in coords:
        cv2.line(img, coords['right hip socket'], coords['right knee'], (255, 255, 255), line_weight)
    if 'right hip socket' in coords and 'hip' in coords:
        cv2.line(img, coords['right hip socket'], coords['hip'], (255, 255, 255), line_weight)
    if 'hip' in coords and 'spine' in coords:
        cv2.line(img, coords['hip'], coords['spine'], (255, 255, 255), line_weight)

    # Draw the joints
    for label, (x, y) in coords.items():
        color = color_map.get(label, (255, 255, 255))  # Default to white if label is not found
        cv2.circle(img, (x, y), point_size, color, -1)

    return img

# Directory containing the images
image_dir = './leg-hip-annotations/images'
output_dir = './leg-hip-annotations/source'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each image in the directory
for image_file in os.listdir(image_dir):
    if image_file.endswith('.png') or image_file.endswith('.jpg'):
        image_path = os.path.join(image_dir, image_file)
        print(f"Processing file: {image_path}")  # Debugging line
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Failed to read image: {image_path}")  # Debugging line
            continue
        
        # Filter the DataFrame for the current image file
        df_image = df[df['file'] == image_file]
        
        if df_image.empty:
            print(f"No matching entries found in CSV for: {image_file}")  # Debugging line
            continue
        
        # Convert image to RGB (OpenCV loads images in BGR by default)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Draw the stick figure on the image
        stick_figure_image = draw_stick_figure(image_rgb, df_image)
        
        # Save the result
        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, cv2.cvtColor(stick_figure_image, cv2.COLOR_RGB2BGR))


# Directory paths
image_dir = './leg-hip-annotations/images'
source_dir = './leg-hip-annotations/source'

# Initialize a list to hold the JSON objects
json_list = []

# Iterate through the image directory
for image_file in os.listdir(image_dir):
    if image_file.endswith('.png') or image_file.endswith('.jpg'):
        # Construct the paths for the source and target images
        source_path = os.path.join(source_dir, image_file)
        target_path = os.path.join(image_dir, image_file)

        # Create a dictionary for the JSON object
        json_object = {
            "source": source_path,
            "target": target_path,
            "prompt": "an x-ray of the legs and hips from the frontal perspective"
        }

        # Add the JSON object to the list
        json_list.append(json_object)

# Convert the list to a JSON formatted string
json_output = json.dumps(json_list, indent=4)

# Save the JSON formatted string to a file
output_file = './leg-hip-annotations/output.json'
with open(output_file, 'w') as f:
    f.write(json_output)

print(f"JSON output saved to {output_file}")
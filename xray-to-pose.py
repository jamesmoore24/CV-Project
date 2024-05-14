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

# Directory paths
image_dir = './leg-hip-annotations/target'
source_dir = './leg-hip-annotations/source'

# Ensure the source directory exists
os.makedirs(source_dir, exist_ok=True)

# Initialize a list to hold the JSON objects
json_list = []

# Iterate through the image directory
for image_file in os.listdir(image_dir):
    if image_file.endswith('.png') or image_file.endswith('.jpg'):
        # Construct the paths for the source and target images
        source_path = os.path.join(source_dir, image_file)
        target_path = os.path.join(image_dir, image_file)

        # Create relative paths to drop the './leg-hip-annotations/' prefix
        relative_source_path = os.path.relpath(source_path, start='./leg-hip-annotations')
        relative_target_path = os.path.relpath(target_path, start='./leg-hip-annotations')

        # Create a dictionary for the JSON object
        json_object = {
            "source": relative_source_path,
            "target": relative_target_path,
            "prompt": "an x-ray of the legs and hips from the frontal perspective"
        }

        # Add the JSON object to the list
        json_list.append(json_object)

# Save each JSON object on a new line in the file
output_file = './leg-hip-annotations/prompt.json'
with open(output_file, 'w') as f:
    for json_object in json_list:
        f.write(json.dumps(json_object) + '\n')

print(f"JSON output saved to {output_file}")
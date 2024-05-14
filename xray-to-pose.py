import pandas as pd
import cv2
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

def adjust_coordinates(x, y, original_dim, target_dim):
    """
    Adjust coordinates from original dimensions to target dimensions.
    """
    original_width, original_height = original_dim
    target_width, target_height = target_dim
    new_x = int(x * target_width / original_width)
    new_y = int(y * target_height / original_height)
    return new_x, new_y

def rotate_and_adjust_coordinates(annotations, original_dim, target_dim):
    """
    Rotate coordinates 90 degrees clockwise and adjust from original dimensions to target dimensions.
    """
    original_width, original_height = original_dim
    target_width, target_height = target_dim

    rotated_coords = annotations.copy()
    rotated_coords['x'] = annotations['y'].apply(lambda y: target_width - adjust_coordinates(0, y, (original_height, original_width), target_dim)[1])
    rotated_coords['y'] = annotations['x'].apply(lambda x: adjust_coordinates(x, 0, (original_width, original_height), target_dim)[0])

    return rotated_coords

def process_image(image_path, annotations, image_output_dir, stick_figure_output_dir):
    # Load the image
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Check the dimensions and rotate if necessary
    if width == 373 and height == 454:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        annotations = rotate_and_adjust_coordinates(annotations, (454, 373), (373, 454))
        height, width = width, height

    # Rotate and rescale coordinates for images with dimensions 2880x2304
    if width == 2304 and height == 2880:
        original_dim = (2304, 2880)
        target_dim = (454, 373)

        # Rotate the image
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        # Rescale the image
        image = cv2.resize(image, target_dim)

        # Adjust the coordinates
        annotations = rotate_and_adjust_coordinates(annotations, original_dim, target_dim)

        height, width = target_dim

    # Save the processed (resized and/or rotated) image
    processed_image_path = os.path.join(image_output_dir, os.path.basename(image_path))
    cv2.imwrite(processed_image_path, image)

    # Draw the stick figure
    stick_figure = draw_stick_figure(image, annotations)

    # Save the stick figure image
    stick_figure_image_path = os.path.join(stick_figure_output_dir, os.path.basename(image_path))
    cv2.imwrite(stick_figure_image_path, stick_figure)

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

# Ensure the output directories exist
image_output_dir = './leg-hip-annotations/new_images'
stick_figure_output_dir = './leg-hip-annotations/new_stick_figures'
os.makedirs(image_output_dir, exist_ok=True)
os.makedirs(stick_figure_output_dir, exist_ok=True)

# Process all images
images_dir = './leg-hip-annotations/target'
for file in os.listdir(images_dir):
    if file.endswith('.jpg'):
        image_path = os.path.join(images_dir, file)
        annotations = df[df['file'] == file].copy()
        process_image(image_path, annotations, image_output_dir, stick_figure_output_dir)


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

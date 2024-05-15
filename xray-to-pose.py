import pandas as pd
import cv2
import numpy as np
import os
import json
import random

# Ensure the output directories exist
image_output_dir = './leg-hip-annotations/target'
stick_figure_output_dir = './leg-hip-annotations/source'
stick_figure_original_output_dir = './stick_figures_original'
os.makedirs(image_output_dir, exist_ok=True)
os.makedirs(stick_figure_output_dir, exist_ok=True)
os.makedirs(stick_figure_original_output_dir, exist_ok=True)

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
csv_file = './point_annotations.csv'
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

def process_image(image_path, annotations, image_output_dir, stick_figure_output_dir, stick_figure_original_output_dir):
    # Load the image
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Draw the stick figure on the original image and save
    overlay_image = overlay_stick_figure_on_original(image, annotations)
    stick_figure_original_image_path = os.path.join(stick_figure_original_output_dir, os.path.basename(image_path))
    cv2.imwrite(stick_figure_original_image_path, overlay_image)
        
    # Draw stick figure for target image
    stick_figure_image = draw_stick_figure(image, annotations)

    # Calculate the bounding box containing all points with margin
    margin = 50  # Adjust the margin as needed
    min_x, min_y = annotations[['x', 'y']].min() - margin
    max_x, max_y = annotations[['x', 'y']].max() + margin

    # Calculate the crop box ensuring a 1:1 aspect ratio
    crop_width = max_x - min_x
    crop_height = max_y - min_y
    if crop_width > crop_height:
        diff = crop_width - crop_height
        diff1 = diff // 2
        diff2 = diff - diff1
        min_y = min_y - diff1
        max_y = max_y + diff2
        crop_height = max_y - min_y
    else:
        diff = crop_height - crop_width
        diff1 = diff // 2
        diff2 = diff - diff1
        min_x = min_x - diff1
        max_x = max_x + diff2
        crop_width = max_x - min_x
    
    padding_margin = 50
    # Calculate padding if the crop box exceeds image dimensions
    top_pad = max(0, -min_y + padding_margin)
    bottom_pad = max(0, max_y + padding_margin - height)
    left_pad = max(0, -min_x + padding_margin)
    right_pad = max(0, max_x + padding_margin - width)

    # Pad the image if necessary
    if top_pad > 0 or bottom_pad > 0 or left_pad > 0 or right_pad > 0:
        image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        stick_figure_image = cv2.copyMakeBorder(stick_figure_image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        height, width = image.shape[:2]

    min_x += left_pad
    max_x += left_pad
    min_y += top_pad
    max_y += top_pad

    # Crop the image and annotations
    image_cropped = image[min_y:max_y, min_x:max_x]
    stick_figure_image_cropped = stick_figure_image[min_y:max_y, min_x:max_x]

    # Resize the images
    image_resized = cv2.resize(image_cropped, (256, 256))
    stick_figure_image_resized = cv2.resize(stick_figure_image_cropped, (256, 256))

    def rotate_image(image, angle):
        """
        Rotate image by the given angle.
        """
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return image
    
    # Save rotated and blurred images
    for angle in [0, 90, 180, 270]:
        # Rotate images
        image_rotated = rotate_image(image_resized, angle)
        stick_figure_image_rotated = rotate_image(stick_figure_image_resized, angle)

        # Save rotated images
        processed_image_rotated_path = os.path.join(image_output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_rot{angle}.jpg")
        stick_figure_image_rotated_path = os.path.join(stick_figure_output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_rot{angle}.jpg")
        cv2.imwrite(processed_image_rotated_path, image_rotated)
        cv2.imwrite(stick_figure_image_rotated_path, stick_figure_image_rotated)

        # # Apply motion blur
        kernel_size = 15
        kernel_motion_blur = np.zeros((kernel_size, kernel_size))
        kernel_motion_blur[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel_motion_blur = kernel_motion_blur / kernel_size

        stick_figure_image_blurred = cv2.filter2D(stick_figure_image_rotated, -1, kernel_motion_blur)

        # Save blurred images
        processed_image_blurred_path = os.path.join(image_output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_rot{angle}_blur.jpg")
        stick_figure_image_blurred_path = os.path.join(stick_figure_output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_rot{angle}_blur.jpg")
        # blurred posemap should still map to same image
        cv2.imwrite(processed_image_blurred_path, image_rotated)
        cv2.imwrite(stick_figure_image_blurred_path, stick_figure_image_blurred)

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

    line_colors = {
        ('left hip socket', 'left knee'): (0, 255, 255),    # Cyan
        ('left hip socket', 'hip'): (128, 0, 128),          # Purple
        ('right hip socket', 'right knee'): (255, 20, 147), # Deep Pink
        ('right hip socket', 'hip'): (75, 0, 130),          # Indigo
        ('hip', 'spine'): (0, 128, 128)                     # Teal
    }

    # Draw the connections based on the specified rules with unique colors
    if 'left hip socket' in coords and 'left knee' in coords:
        cv2.line(img, coords['left hip socket'], coords['left knee'], line_colors[('left hip socket', 'left knee')], line_weight)
    if 'left hip socket' in coords and 'hip' in coords:
        cv2.line(img, coords['left hip socket'], coords['hip'], line_colors[('left hip socket', 'hip')], line_weight)
    if 'right hip socket' in coords and 'right knee' in coords:
        cv2.line(img, coords['right hip socket'], coords['right knee'], line_colors[('right hip socket', 'right knee')], line_weight)
    if 'right hip socket' in coords and 'hip' in coords:
        cv2.line(img, coords['right hip socket'], coords['hip'], line_colors[('right hip socket', 'hip')], line_weight)
    if 'hip' in coords and 'spine' in coords:
        cv2.line(img, coords['hip'], coords['spine'], line_colors[('hip', 'spine')], line_weight)

    # Draw the joints
    for label, (x, y) in coords.items():
        color = color_map.get(label, (255, 255, 255))  # Default to white if label is not found
        cv2.circle(img, (x, y), point_size, color, -1)

    return img

def overlay_stick_figure_on_original(image, df):
    # Create a copy of the original image
    img = image.copy()
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

    line_colors = {
        ('left hip socket', 'left knee'): (0, 255, 255),    # Cyan
        ('left hip socket', 'hip'): (128, 0, 128),          # Purple
        ('right hip socket', 'right knee'): (255, 20, 147), # Deep Pink
        ('right hip socket', 'hip'): (75, 0, 130),          # Indigo
        ('hip', 'spine'): (0, 128, 128)                     # Teal
    }

    # Draw the connections based on the specified rules with unique colors
    if 'left hip socket' in coords and 'left knee' in coords:
        cv2.line(img, coords['left hip socket'], coords['left knee'], line_colors[('left hip socket', 'left knee')], line_weight)
    if 'left hip socket' in coords and 'hip' in coords:
        cv2.line(img, coords['left hip socket'], coords['hip'], line_colors[('left hip socket', 'hip')], line_weight)
    if 'right hip socket' in coords and 'right knee' in coords:
        cv2.line(img, coords['right hip socket'], coords['right knee'], line_colors[('right hip socket', 'right knee')], line_weight)
    if 'right hip socket' in coords and 'hip' in coords:
        cv2.line(img, coords['right hip socket'], coords['hip'], line_colors[('right hip socket', 'hip')], line_weight)
    if 'hip' in coords and 'spine' in coords:
        cv2.line(img, coords['hip'], coords['spine'], line_colors[('hip', 'spine')], line_weight)

    # Draw the joints
    for label, (x, y) in coords.items():
        color = color_map.get(label, (255, 255, 255))  # Default to white if label is not found
        cv2.circle(img, (x, y), point_size, color, -1)

    return img

# Test suite
def check_square_images(directory):
    """
    Check if all images in the given directory are square.
    Prints out the result for each image.
    """
    all_square = True

    for file in os.listdir(directory):
        if file.endswith('.jpg') or file.endswith('.png'):
            image_path = os.path.join(directory, file)
            image = cv2.imread(image_path)
            height, width = image.shape[:2]

            if (height, width) != (256, 256):
                all_square = False
                print(f"Image {file} is not square: {width}x{height}")
    return all_square

def count_images(directory):
    """
    Count the number of images in the given directory.
    """
    count = 0
    for file in os.listdir(directory):
        if file.endswith('.jpg') or file.endswith('.png'):
            count += 1
    return count

json_list = []

prompts = [
    "an x-ray of the legs and hips from the frontal perspective"
]

# Process all images in the original_images directory
original_images_dir = './original_images'
for file in os.listdir(original_images_dir):
    if file.endswith('.jpg') or file.endswith('.png'):
        image_path = os.path.join(original_images_dir, file)
        image_annotations = df[df['file'] == file]
        process_image(image_path, image_annotations, image_output_dir, stick_figure_output_dir, stick_figure_original_output_dir)



images_dir = 'leg-hip-annotations/source'
for file in os.listdir(images_dir):
    if file.endswith('.jpg') or file.endswith('.png'):
        # Create relative paths to drop the './leg-hip-annotations/' prefix
        relative_source_path = os.path.relpath(os.path.join(stick_figure_output_dir, file), start='./leg-hip-annotations')
        relative_target_path = os.path.relpath(os.path.join(image_output_dir, file), start='./leg-hip-annotations')

        # Randomly sample a prompt from the list
        prompt = random.choice(prompts)

        # Create a dictionary for the JSON object
        json_object = {
            "source": relative_source_path,
            "target": relative_target_path,
            "prompt": prompt
        }

        # Add the JSON object to the list
        json_list.append(json_object)

# Convert the list to a JSON formatted string
json_output = json.dumps(json_list, indent=4)

# Save the JSON formatted string to a file
output_file = './leg-hip-annotations/prompt.json'
with open(output_file, 'w') as f:
    for json_object in json_list:
        json.dump(json_object, f)
        f.write('\n')

print(f"JSON output saved to {output_file}")
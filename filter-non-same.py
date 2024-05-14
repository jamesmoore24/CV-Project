import os
import cv2

def remove_non_matching_images(directory, expected_width, expected_height):
    """
    Removes images in the specified directory that do not have the expected dimensions.
    """
    for file in os.listdir(directory):
        if file.endswith('.jpg') or file.endswith('.png'):
            image_path = os.path.join(directory, file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read {image_path}, skipping.")
                continue
            height, width = image.shape[:2]
            if (width, height) != (expected_width, expected_height):
                os.remove(image_path)
                print(f"Removed {image_path} with dimensions ({width}, {height})")

# Directories to check
image_output_dir = './leg-hip-annotations/target'
stick_figure_output_dir = './leg-hip-annotations/source'

# Expected dimensions
expected_width = 454
expected_height = 373

# Remove non-matching images in both directories
remove_non_matching_images(image_output_dir, expected_width, expected_height)
remove_non_matching_images(stick_figure_output_dir, expected_width, expected_height)

print("Cleanup complete.")

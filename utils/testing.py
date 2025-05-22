import numpy as np
import cv2  # Import OpenCV
import os
import glob
import time  # Import the time module to measure runtime
from concurrent.futures import ThreadPoolExecutor

def get_image_paths_from_folder(folder_path):
    """
    Retrieves a list of image paths (AVIF and JPG) from the specified folder.

    Parameters:
    folder_path (str): The path to the folder containing images.

    Returns:
    list of str: List of full image paths.
    """
    valid_extensions = ['*.avif', '*.jpg']
    image_paths = []
    
    for ext in valid_extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
    
    return image_paths

def load_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    return img

def concat_images_horizontally(image_paths, output_path):
    with ThreadPoolExecutor(max_workers=20) as executor:
        images = list(executor.map(load_image, image_paths))
    total_width = sum(img.shape[1] for img in images)
    max_height = max(img.shape[0] for img in images)
    
    # Preallocate the concatenated image array
    concatenated_image = np.zeros((max_height, total_width, 3), dtype=np.uint8)

    # Fill in the pixel data for the concatenated image
    current_x = 0
    for img in images:
        height = img.shape[0]
        concatenated_image[:height, current_x:current_x + img.shape[1], :] = img
        current_x += img.shape[1]
    
    # Save the concatenated image using OpenCV
    cv2.imwrite(output_path, concatenated_image)

    end_time = time.time()  # End time measurement
    print(f"Time taken to concatenate images horizontally: {end_time - start_time:.2f} seconds")

def concat_images_vertically(image_paths, output_path):
    """
    Concatenates images vertically and saves the result.

    Parameters:
    image_paths (list of str): List of image paths to concatenate.
    output_path (str): Path to save the concatenated image.
    """
    start_time = time.time()  # Start time measurement

    # Load images concurrently using OpenCV
    with ThreadPoolExecutor(max_workers=20) as executor:
        images = list(executor.map(load_image, image_paths))

    # Calculate maximum width and total height
    max_width = max(img.shape[1] for img in images)   # Max width
    total_height = sum(img.shape[0] for img in images)  # Sum heights
    
    # Preallocate the concatenated image array
    concatenated_image = np.zeros((total_height, max_width, 3), dtype=np.uint8)

    # Fill in the pixel data for the concatenated image
    current_y = 0
    for img in images:
        width = img.shape[1]
        height = img.shape[0]
        concatenated_image[current_y:current_y + height, :width, :] = img
        current_y += height

    # Save the concatenated image using OpenCV
    cv2.imwrite(output_path, concatenated_image)

    end_time = time.time()  # End time measurement
    print(f"Time taken to concatenate images vertically: {end_time - start_time:.2f} seconds")


# Example usage
if __name__ == "__main__":
    # Replace with your folder path containing AVIF and JPG images
    folder_path = '/dataset/AIC2024/pumkin_dataset/1/frames/autoshot/Keyframes_L13/keyframes/L13_V001'
    image_paths = get_image_paths_from_folder(folder_path)  # Limit to 100 images for testing

    output_path_horizontal = 'concatenated_horizontal.jpg'
    output_path_vertical = 'concatenated_vertical.jpg'

    # Concatenate images horizontally and measure runtime
    concat_images_horizontally(image_paths, output_path_horizontal)

    # Concatenate images vertically and measure runtime
    concat_images_vertically(image_paths, output_path_vertical)

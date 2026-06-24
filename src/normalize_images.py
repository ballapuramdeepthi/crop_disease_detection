import os
import cv2
import numpy as np

input_dir = "processed_images"
output_dir = "normalized_images"

os.makedirs(output_dir, exist_ok=True)

for image_name in os.listdir(input_dir):
    image_path = os.path.join(input_dir, image_name)

    image = cv2.imread(image_path)

    if image is not None:
        normalized_image = image.astype(np.float32) / 255.0

        save_path = os.path.join(output_dir, image_name)
        np.save(save_path, normalized_image)

print("Image normalization completed successfully.")
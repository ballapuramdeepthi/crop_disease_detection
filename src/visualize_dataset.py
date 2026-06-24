import os
import random
from PIL import Image
import matplotlib.pyplot as plt

DATASET_DIR = "plantvillage dataset/color"

folders = [f for f in os.listdir(DATASET_DIR)
           if os.path.isdir(os.path.join(DATASET_DIR, f))]

fig, axes = plt.subplots(4, 4, figsize=(16, 16))

for ax, folder in zip(axes.ravel(), folders[:16]):

    folder_path = os.path.join(DATASET_DIR, folder)

    images = os.listdir(folder_path)

    random_image = random.choice(images)

    image_path = os.path.join(folder_path, random_image)

    image = Image.open(image_path)

    ax.imshow(image)
    ax.set_title(folder.replace("_", "\n"), fontsize=8)
    ax.axis("off")

plt.suptitle("Random Crop Disease Samples", fontsize=18)
plt.tight_layout()
plt.savefig("crop_samples_grid.png")
plt.show()
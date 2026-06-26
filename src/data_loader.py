import tensorflow as tf

# Dataset path
dataset_path = "plantvillage dataset/color"

# Training Dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(224, 224),
    batch_size=32
)

# Validation Dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(224, 224),
    batch_size=32
)

print("Train Dataset Created")
print("Validation Dataset Created")

# Get Class Names
class_names = train_ds.class_names

print("\nClasses:")
for cls in class_names:
    print(cls)

print(f"\nTotal Classes: {len(class_names)}")

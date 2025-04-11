import os
import random
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_DIR = 'emotion_dataset_cleaned'
TARGET_SIZE = 30
IMG_SIZE = (224, 224)

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

for emotion in os.listdir(DATASET_DIR):
    emotion_path = os.path.join(DATASET_DIR, emotion)
    images = [f for f in os.listdir(emotion_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    current_count = len(images)
    if current_count >= TARGET_SIZE:
        print(f"{emotion}: {current_count} images — OK")
        continue

    print(f"{emotion}: {current_count} images — augmenting to {TARGET_SIZE}")
    needed = TARGET_SIZE - current_count
    image_files = random.choices(images, k=needed)

    for i, img_name in enumerate(image_files):
        img_path = os.path.join(emotion_path, img_name)
        img = Image.open(img_path).resize(IMG_SIZE)
        img_array = np.expand_dims(np.array(img), axis=0)

        aug_iter = datagen.flow(img_array, batch_size=1)
        aug_img = next(aug_iter)[0].astype('uint8')
        aug_img_pil = Image.fromarray(aug_img)

        save_name = f"aug_{i}_{img_name}"
        save_path = os.path.join(emotion_path, save_name)
        aug_img_pil.save(save_path)

print("\n Augmentation complete!")

import os
import shutil
import random

# Parameters
SOURCE_DIR = "emotion_dataset_cleaned"
OUTPUT_DIR = "emotion_dataset_split_aug_30"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 99

random.seed(SEED)

splits = ['train', 'val', 'test']
categories = os.listdir(SOURCE_DIR)

for split in splits:
    for category in categories:
        os.makedirs(os.path.join(OUTPUT_DIR, split, category), exist_ok=True)

for category in categories:
    files = os.listdir(os.path.join(SOURCE_DIR, category))
    random.shuffle(files)

    total = len(files)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    split_files = {
        'train': files[:train_end],
        'val': files[train_end:val_end],
        'test': files[val_end:]
    }

    for split, file_list in split_files.items():
        for file_name in file_list:
            src = os.path.join(SOURCE_DIR, category, file_name)
            dst = os.path.join(OUTPUT_DIR, split, category, file_name)
            shutil.copy2(src, dst)

print("Dataset successfully split and saved in:", OUTPUT_DIR)

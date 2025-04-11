import os
from duckduckgo_search import DDGS
import requests
from PIL import Image
from io import BytesIO

emotions = ["anger", "contempt", "disgust", "enjoyment", "fear", "sadness", "surprise"]
base_dir = "emotion_dataset"
os.makedirs(base_dir, exist_ok=True)

def download_image(url, save_path):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        content_type = response.headers.get('Content-Type', '')
        if 'image' not in content_type:
            print(f"Skipped (not an image): {url}")
            return False
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img.save(save_path)
        print(f"Saved: {save_path}")
        return True
    except Exception as e:
        print(f"Failed to save {url} | Reason: {e}")
        return False

def scrape_emotion_images(emotion, max_images=30):
    save_dir = os.path.join(base_dir, emotion)
    os.makedirs(save_dir, exist_ok=True)
    query = f"human facial expression {emotion}"
    count = 0
    with DDGS() as ddgs:
        for result in ddgs.images(query, max_results=max_images*2):
            if count >= max_images:
                break
            url = result.get("image")
            if url:
                filename = os.path.join(save_dir, f"{emotion}_{count}.jpg")
                if download_image(url, filename):
                    count += 1
    print(f"{emotion}: {count} images saved.")

for emotion in emotions:
    scrape_emotion_images(emotion)

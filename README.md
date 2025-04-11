# Facial Emotion Classifier

A simple CNN-based app to classify facial emotions using deep learning and Streamlit.

## Features

- Detects 7 emotions: **anger, contempt, disgust, enjoyment, fear, sadness, surprise**
- Upload a face image and get predictions + confidence chart
- Built with TensorFlow + Streamlit

## 🚀 Try It Live

👉 [Hugging Face Space](https://huggingface.co/spaces/maulanaibrohim/facial-emotion-classifier)

## 🛠️ Run Locally

```bash
git clone https://github.com/your-username/face-emotion-classifier.git
cd face-emotion-classifier
pip install -r requirements.txt
streamlit run app.py
```

## Files

augment_data.py – Adds image augmentations to expand the dataset.
train_model.py – Trains the CNN model on the dataset.
split_dataset.py – Splits data into train/val/test sets.
scrape_image.py – Collects facial images (e.g., via web scraping).
evaluate_model.py – Evaluates model performance on test data.

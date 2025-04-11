import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
from datetime import datetime

BASE_DIR = "emotion_dataset_split_aug_30"
IMG_SIZE = (128, 128)
BATCH_SIZE = 16

test_gen = ImageDataGenerator(rescale=1./255)
test_data = test_gen.flow_from_directory(
    os.path.join(BASE_DIR, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

model = load_model("face_model_aug_30.h5")

predictions = model.predict(test_data)
y_pred = np.argmax(predictions, axis=1)
y_true = test_data.classes
class_labels = list(test_data.class_indices.keys())
filenames = test_data.filenames

report = classification_report(y_true, y_pred, target_names=class_labels)
print("Classification Report:")
print(report)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_filename = f"classification_report_aug_30_{timestamp}.txt"
with open(report_filename, "w", encoding="utf-8") as f:
    f.write("Classification Report:\n\n")
    f.write(report)
print(f"Classification report saved as '{report_filename}'")

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()

cm_filename = "confusion_matrix.png"
plt.savefig(cm_filename)
print(f"Confusion matrix saved as '{cm_filename}'")
plt.show()

print("Evaluation complete!")
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score

# 1. Load images and labels
def load_image_data(root_dir, image_size=(64, 64)):
    X, y = [], []
    class_names = sorted(os.listdir(root_dir))
    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            fpath = os.path.join(cls_dir, fname)
            try:
                img = Image.open(fpath).convert('RGB')
                img = img.resize(image_size)
                arr = np.array(img).flatten() / 255.0
                X.append(arr)
                y.append(idx)
            except Exception as e:
                print(f"Skipping {fpath}: {e}")
    return np.vstack(X), np.array(y), class_names
    
DATA_DIR = "Trash Data/images" #Training images path

X, y, class_names = load_image_data(DATA_DIR)
print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features each.")

# 2. Train/test split (binary classification assumed; adjust if multiclass)
if len(class_names) != 2:
    raise ValueError("This script assumes exactly two classes for ROC/KS. Found: " + str(class_names))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Fit MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                    max_iter=200, random_state=42)
clf.fit(X_train, y_train)

# 4. Predict probabilities and compute ROC/AUC
y_score = clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
print(f"AUC: {roc_auc:.4f}")

# 5. Compute KS statistic
ks_statistic = np.max(np.abs(tpr - fpr))
print(f"KS Statistic: {ks_statistic:.4f}")

# 6. Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Chance level')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1) Load images and labels from directory structure:
def load_data(root_dir, img_size=(64,64)):
    X, y = [], []
    classes = sorted(os.listdir(root_dir))
    for label in classes:
        class_dir = os.path.join(root_dir, label)
        if not os.path.isdir(class_dir): continue
        for fn in os.listdir(class_dir):
            path = os.path.join(class_dir, fn)
            try:
                img = Image.open(path).convert("RGB").resize(img_size)
                arr = np.array(img).flatten()   # → (64*64*3,)
                X.append(arr)
                y.append(label)
            except Exception:
                continue
    return np.vstack(X), np.array(y)

X, y = load_data("Trash Data/images")
print(f"Loaded {X.shape[0]} samples, feature dim = {X.shape[1]}")

# 2) Encode string labels to integers
le = LabelEncoder()
y_enc = le.fit_transform(y) 

# 3) Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.3, random_state=42, stratify=y_enc
)

# 4) Build a pipeline: scaling → PCA dim‐reduction → MLP classifier
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=50)),      # reduce to 50 features
    ("clf", MLPClassifier(
        hidden_layer_sizes=(100,),
        activation='relu',
        solver='adam',
        max_iter=300,
        random_state=42
    )),
])

# 5) Hyperparameter search
param_grid = {
    "pca__n_components": [30, 50, 100],
    "clf__hidden_layer_sizes": [(50,), (100,), (100,50)],
}
grid = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1, scoring="accuracy")
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)

# 6) Evaluate on test set
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

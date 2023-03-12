import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load
import os


# Load data
labels = np.load("data/labels.npy")
features = np.load("data/features.npy")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train SVM model
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

# Evaluate model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save SVM model to file
MODEL_PATH = "model/"
if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)
dump(clf, "model/svm_model.joblib")

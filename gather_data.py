import os
import cv2
import numpy as np
from pynput import keyboard

# Define constants
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
DATA_DIR = "data/"
LABELS_FILE = DATA_DIR + "labels.npy"
FEATURES_FILE = DATA_DIR + "features.npy"
ALPHABET = [chr(i) for i in range(ord('A'), ord('Z')+1)
            if chr(i) != 'J' and chr(i) != 'Z']

# Create data directory if it does not exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Load existing data
labels = np.load(LABELS_FILE) if os.path.exists(LABELS_FILE) else np.array([])
features = np.load(FEATURES_FILE) if os.path.exists(FEATURES_FILE) else np.empty((0, 128))

# Preprocess image
def preprocess(image):
    return image
  
# Define feature extraction function
def extract_features(image):
    # Preprocess image
    # TODO: Implement preprocessing
    # Apply feature extraction technique
    # TODO: Implement feature extraction
    return np.random.rand(128)  # Replace with actual feature vector

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

# Define key press callback function
def on_press(key):
    global labels, features, frame
    try:
        letter = key.char.upper()
        if letter in ALPHABET:
            label = letter
            features = np.vstack((features, extract_features(frame)))
            labels = np.append(labels, label)
            print("Data saved for letter", label)
            # Add caption
            cv2.putText(frame, label, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    except AttributeError:
        pass


# Start key press listener
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Start capturing and processing data
while True:
    # Capture frame from camera
    ret, frame = cap.read()
    if not ret:
        break

    # Display frame
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop key press listener and release resources
listener.stop()
cap.release()
cv2.destroyAllWindows()

# Save data to disk
np.save(LABELS_FILE, labels)
np.save(FEATURES_FILE, features)

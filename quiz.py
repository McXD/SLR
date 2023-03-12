import cv2
import numpy as np
import time
import random
from sklearn import svm
from joblib import load

# Preprocess image
def preprocess_image(image):
    return image
  
# Load k-means clustering model and bag of visual words dictionary from disk
kmeans = np.load("data/kmeans.npy")
codebook = np.load("data/codebook.npy")

# Define function for extracting bag of visual words features from an image
def extract_features(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize image to (128, 128)
    resized = cv2.resize(gray, (128, 128))
    # Extract SIFT features from image
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(resized, None)
    # Assign each feature to the nearest codebook cluster
    clusters = kmeans.predict(descriptors)
    # Convert the cluster assignments to a bag of visual words histogram
    histogram = vectorizer.transform([" ".join([str(x) for x in clusters])])
    # Convert the histogram to a 1D array
    feature_vector = histogram.toarray()[0]
    # Normalize the feature vector
    # TODO
    feature_vector /= np.linalg.norm(feature_vector)
    return feature_vector
  
# Load SVM model
clf = load("model/svm_model.joblib")

# Load letter-to-label mapping
LABELS = {
    "A": "A",
    "B": "B",
    "C": "C",
    "D": "D",
    "E": "E",
    "F": "F",
    "G": "G",
    "H": "H",
    "I": "I",
    "K": "K",
    "L": "L",
    "M": "M",
    "N": "N",
    "O": "O",
    "P": "P",
    "Q": "Q",
    "R": "R",
    "S": "S",
    "T": "T",
    "U": "U",
    "V": "V",
    "W": "W",
    "X": "X",
    "Y": "Y"
}

# Start camera
cap = cv2.VideoCapture(0)

# Define font for captions
font = cv2.FONT_HERSHEY_SIMPLEX

# Define round time limit in seconds
round_time_limit = 6

# Define list of letters to use for quiz
letters = list(LABELS.keys())

# Start game loop
score = 0
round_num = 1
while True:
    # Prompt user with a random letter
    letter = random.choice(letters)
  # Add prompt and score captions to frame
    prompt = "Round " + str(round_num) + " - Letter: " + letter
    score_caption = "Score: " + str(score)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Reset round timer
    round_start_time = time.time()

    # Enter round loop
    round_correct = False
    while time.time() - round_start_time < round_time_limit:
        # Read frame from camera
        ret, frame = cap.read()

        # Preprocess frame
        frame = preprocess_image(frame)

        # Extract features from frame
        features = extract_features(frame)

        # Make prediction using SVM model
        label = clf.predict([features])[0]

        # Get corresponding letter for label
        predicted_letter = LABELS.get(label, "Unknown")

        # Check if predicted letter is correct
        if predicted_letter == letter:
            round_correct = True
            break

        # Add prompt and predicted letter captions to frame
        cv2.putText(frame, prompt, (50, 50), font, 1, (255, 255, 255), 2)
        cv2.putText(frame, score_caption, (50, 100),
                    font, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Predicted: " + predicted_letter,
                    (50, 150), font, 1, (0, 0, 255), 2)
        time_caption = "Round time left: {:.0f}s".format(round(round_time_limit - (time.time() - round_start_time)))
        cv2.putText(frame, time_caption, (50, 200), font, 1, (0, 0, 255), 2)


        # Show frame
        cv2.imshow("Quiz", frame)

        # Check for key press
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    # Update score and round number
    if round_correct:
        score += 1
        
    round_num += 1

    # Check if game is over
    if round_num > 10:
        break

# Print final score and release camera
print("Final score:", score)
cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import os

# Create directories to save images
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("data/images"):
    os.makedirs("data/images")

# Initialize camera object
cap = cv2.VideoCapture(0)
# Set the frame size to 640x480 pixels
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

# Define image size
img_size = (128, 128)

# Initialize counter for number of data files collected
num_files = len(os.listdir('data/images'))

while True:
    # Capture frame from camera
    ret, frame = cap.read()

    # Display instructions on frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Press any letter key (excluding J and Z) to save an image",
                (10, 50), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Press 'q' to quit",
                (10, 100), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{num_files} images collected",
                (10, 150), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Display frame
    cv2.imshow("Camera", frame)

    # Check for key press
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key >= ord('a') and key <= ord('z') and key != ord('j') and key != ord('z'):
        # Save image with corresponding label
        letter = chr(key).upper()
        filename = os.path.join("data", "images", f"{letter}_{num_files}.jpg")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, img_size)
        cv2.imwrite(filename, gray)
        # Increment counter for number of data files collected
        num_files += 1

# Release camera and close window
cap.release()
cv2.destroyAllWindows()

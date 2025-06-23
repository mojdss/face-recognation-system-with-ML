import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from joblib import dump, load
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
from skimage.transform import resize

# Load the Olivetti faces dataset
data = fetch_olivetti_faces()
X = data.images
y = data.target

# Flatten the images for the model
n_samples, h, w = X.shape
X = X.reshape((n_samples, h * w))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM model
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Save the model
dump(model, 'olivetti_svm_model.joblib')

# Load the trained model
model = load('olivetti_svm_model.joblib')

def prepare_image(image):
    image_resized = resize(image, (64, 64), anti_aliasing=True)
    image_flattened = image_resized.reshape(1, -1)
    return image_flattened

def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = prepare_image(gray_frame)
        prediction = model.predict(image)

        # Display the label on the camera feed
        label = f'Predicted Label: {int(prediction[0])}'
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Camera Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Create the main window
root = tk.Tk()
root.title("Olivetti Faces Prediction")
root.geometry("400x400")

# Create and place widgets
capture_button = ttk.Button(root, text="Capture Image", command=capture_image)
capture_button.pack(pady=20)

# Start the Tkinter main loop
root.mainloop()
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("face_recognition_model.h5")

# Define the path to the test images
data_dir = "C:/Users/mucah/OneDrive/Desktop/python/Face Recognition/Original Images"  # Adjusted path

# Define the target image size
IMG_SIZE = (224, 224)

# Prepare for displaying images and predictions
plt.figure(figsize=(15, 15))  # Adjusted for a grid layout

# Get the list of class names based on directory structure
class_names = os.listdir(data_dir)

# Iterate over each class and its images
for class_name in class_names:
    class_path = os.path.join(data_dir, class_name)

    # Check if the path is a directory
    if not os.path.isdir(class_path):
        continue

    # Get list of images in the class directory and randomly select a subset
    test_images = [f for f in os.listdir(class_path) if f.endswith('.jpg') or f.endswith('.png')]
    selected_images = random.sample(test_images, min(4, len(test_images)))  # Select up to 4 images

    for i, img_name in enumerate(selected_images):
        img_path = os.path.join(class_path, img_name)

        # Load and preprocess the image
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Make a prediction
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_names[predicted_class_index]  # Get predicted class label

        # Extract actual class from the directory name
        actual_class = class_name

        # Check if the prediction matches the actual class
        is_correct = "Correct" if predicted_class == actual_class else "Incorrect"

        # Debug: print image name, predicted class, and actual class
        print(f"Image: {img_name}, Predicted Class: {predicted_class}, Actual Class: {actual_class}, Result: {is_correct}")

        # Display the image and predicted label
        plt.subplot(4, 4, i + 1)
        plt.imshow(img)
        plt.title(f"Predicted: {predicted_class}\nActual: {actual_class}\n{is_correct}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

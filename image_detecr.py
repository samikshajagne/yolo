import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load the YOLO model (replace with your custom model if needed)
model = YOLO('yolov8n.pt')  # Replace 'yolov8n.pt' with the path to your custom weights if needed

# Load the image
image_path = 'images/04.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Perform detection
results = model(image)

# Get the annotated image (with bounding boxes and labels)
annotated_image = results[0].plot()

# Convert BGR (OpenCV format) to RGB (Matplotlib format)
annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

# Display the image using Matplotlib
plt.imshow(annotated_image_rgb)
plt.axis('off')  # Hide axes
plt.show()

# Optionally, save the annotated image
output_path = 'detected_image4.jpg'  # Specify where you want to save the image
cv2.imwrite(output_path, annotated_image)

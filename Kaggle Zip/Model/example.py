# Pothole Detection with ONNX
import cv2
import numpy as np
import onnxruntime as ort

# Load model
session = ort.InferenceSession('model.onnx')

def preprocess(image):
    image = cv2.resize(image, (640, 640))
    image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(image, 0)

def detect_potholes(image_path):
    image = cv2.imread(image_path)
    input_tensor = preprocess(image)
    outputs = session.run(None, {'images': input_tensor})
    return outputs

print("Pothole detector ready! Use detect_potholes('image.jpg')")
import numpy as np
import cv2

def preprocess_image(file):
    # Read file as bytes
    file_bytes = np.frombuffer(file.read(), np.uint8)

    # Decode image
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Image decoding failed")

    # Resize
    img = cv2.resize(img, (224, 224))

    # Normalize
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img
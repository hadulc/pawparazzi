import os
from typing import Tuple
import cv2
import numpy as np
import tensorflow as tf


def load_model()-> tf.keras.Model:
    """
    Loads and returns the Keras model for breed prediction.

    Raises:
        FileNotFoundError: If the model file does not exist.

    Returns:
        tf.keras.Model: Loaded Keras model.
    """
    model_name = "model.keras" if os.getenv("IN_CONTAINER", False) else os.getenv("MODEL_NAME", "model.keras")
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    path = os.path.join(root_path, "models", model_name)
    model = tf.keras.models.load_model(path)
    return model


def preprocess_image(image: np.ndarray, img_size: int = 224) -> np.ndarray:
    """
    Resizes and expands dimensions of the input image for model prediction.

    Args:
        image (np.ndarray): Input image.
        img_size (int): Target size for resizing.

    Returns:
        np.ndarray: Preprocessed image ready for model input.
    """
    resized_img = cv2.resize(src=image, dsize=(img_size, img_size))
    resized_img = tf.expand_dims(input=resized_img, axis=0)
    return resized_img


def predict_breed(
    image: np.ndarray,
    model: tf.keras.Model,
    top_k: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predicts the top K dog breeds for the given image using the provided model.

    Args:
        image (np.ndarray): Input image.
        model (tf.keras.Model): Trained breed classification model.
        top_k (int): Number of top predictions to return.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Top K scores and their corresponding indices.
    """
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    scores = prediction[0]
    top_scores, top_indices = tf.math.top_k(input=scores, k=top_k)
    return top_scores.numpy(), top_indices.numpy()

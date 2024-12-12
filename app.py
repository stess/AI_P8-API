from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from keras.saving import register_keras_serializable
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import io
import base64

def colorize_mask(mask):
    """
    Applique une palette de couleurs au masque.
    Args:
        mask (np.ndarray): Masque prédictif avec des indices de classes.
    Returns:
        np.ndarray: Masque coloré en RGB.
    """
    cmap = plt.get_cmap('tab10')  # Utiliser la palette tab10
    # Normaliser le masque pour qu'il soit entre 0 et 1
    mask_color = cmap(mask / mask.max())
    # Convertir en RGB (0-255)
    mask_color = (mask_color[:, :, :3] * 255).astype(np.uint8)
    return mask_color


MODEL_PATH = "model.keras"

# Initialiser l'application Flask
app = Flask(__name__)


@app.route("/")
def home():
    return "API de prédiction de masque est opérationnelle."


# Lancer l'application
if __name__ == "__main__":
    app.run()

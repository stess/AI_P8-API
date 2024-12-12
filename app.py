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


# Initialiser l'application Flask
app = Flask(__name__)


@app.route("/")
def home():
    return "API de prédiction de masque est opérationnelle."


# Lancer l'application
if __name__ == "__main__":
    app.run(threaded=True)

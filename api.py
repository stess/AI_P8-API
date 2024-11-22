from flask import Flask, request, jsonify, send_file
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import gdown
import os

# URL Google Drive du modèle
MODEL_DRIVE_URL = "https://drive.google.com/uc?id=1WG7lxRpHNoXrJF1VN5KnGWEo5UYZfZcU"
MODEL_PATH = "best_model_albumentations_2.keras"

# Vérifier et télécharger le modèle si nécessaire
if not os.path.exists(MODEL_PATH):
    print("Téléchargement du modèle depuis Google Drive...")
    gdown.download(MODEL_DRIVE_URL, MODEL_PATH, quiet=False)

# Charger le modèle
print("Chargement du modèle...")
model = load_model(MODEL_PATH)

# Dictionnaire des catégories
cats = {
    'void': [0, 1, 2, 3, 4, 5, 6],
    'flat': [7, 8, 9, 10],
    'construction': [11, 12, 13, 14, 15, 16],
    'object': [17, 18, 19, 20],
    'nature': [21, 22],
    'sky': [23],
    'human': [24, 25],
    'vehicle': [26, 27, 28, 29, 30, 31, 32, 33, -1]
}

# Initialiser l'application Flask
app = Flask(__name__)

# Point d'entrée pour prédire le masque
@app.route('/predict', methods=['POST'])
def predict_mask():
    # Vérifier si un fichier image est fourni
    if 'file' not in request.files:
        return jsonify({"error": "Aucune image n'a été fournie"}), 400

    file = request.files['file']

    # Lire l'image
    image = Image.open(file.stream).convert('RGB')
    image = np.array(image)

    # Prétraitement : redimensionner l'image à la taille attendue par le modèle
    image_resized = cv2.resize(image, (512, 256))  # Dimensions attendues par le modèle
    image_resized = image_resized / 255.0  # Normaliser les pixels entre 0 et 1
    image_resized = np.expand_dims(image_resized, axis=0)  # Ajouter une dimension pour le batch

    # Prédire le masque
    predicted_mask = model.predict(image_resized)[0]  # Première image du batch

    # Post-traitement : convertir les probabilités en classes
    predicted_mask = np.argmax(predicted_mask, axis=-1)

    # Redimensionner le masque à la taille originale de l'image d'entrée
    predicted_mask_resized = cv2.resize(predicted_mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Sauvegarder le masque dans un fichier temporaire
    mask_image = Image.fromarray(predicted_mask_resized)
    temp_path = "predicted_mask.png"
    mask_image.save(temp_path)

    # Retourner le fichier prédictif comme réponse
    return send_file(temp_path, mimetype='image/png')

# Point d'entrée pour tester le service
@app.route('/')
def home():
    return "API de prédiction de masque est opérationnelle."

# Lancer l'application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

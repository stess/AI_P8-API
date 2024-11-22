from flask import Flask, request, jsonify, send_file
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from keras.saving import register_keras_serializable
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import gdown
import os

def colorize_mask(mask):
    """
    Applique une palette de couleurs au masque.
    Args:
        mask (np.ndarray): Masque prédictif avec des indices de classes.
    Returns:
        np.ndarray: Masque coloré en RGB.
    """
    cmap = plt.get_cmap('tab10')  # Utiliser la palette tab10
    mask_color = cmap(mask / mask.max())  # Normaliser le masque pour qu'il soit entre 0 et 1
    mask_color = (mask_color[:, :, :3] * 255).astype(np.uint8)  # Convertir en RGB (0-255)
    return mask_color

# URL de téléchargement direct du modèle Google Drive
MODEL_DRIVE_URL = "https://drive.google.com/uc?id=1WG7lxRpHNoXrJF1VN5KnGWEo5UYZfZcU"
MODEL_PATH = "best_model_albumentations_2.keras"

# Fonctions personnalisées
@register_keras_serializable()
def dice_coeff(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f = K.flatten(K.cast(y_pred, 'float32'))
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

@register_keras_serializable()
def dice_loss(y_true, y_pred):
    return 1 - dice_coeff(y_true, y_pred)

@register_keras_serializable()
def total_loss(y_true, y_pred):
    binary_crossentropy_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return binary_crossentropy_loss + (3 * dice_loss(y_true, y_pred))

@register_keras_serializable()
def jaccard_score(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f = K.flatten(K.cast(y_pred, 'float32'))
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# Fonction pour calculer l'IoU (Jaccard) pour chaque classe
@register_keras_serializable()
def calculate_iou(true_mask, pred_mask, num_classes):
    ious = []
    for cls in range(num_classes):
        true_cls = (true_mask == cls).astype(int)
        pred_cls = (pred_mask == cls).astype(int)
        iou = jaccard_score(true_cls.flatten(), pred_cls.flatten())
        ious.append(iou)
    return ious

# Vérifier et télécharger le modèle si nécessaire
if not os.path.exists(MODEL_PATH):
    print("Téléchargement du modèle depuis Google Drive...")
    gdown.download(MODEL_DRIVE_URL, MODEL_PATH, quiet=False)

# Charger le modèle avec la fonction personnalisée
print("Chargement du modèle...")
model = load_model(MODEL_PATH, custom_objects={
    "dice_coeff": dice_coeff,
    "dice_loss": dice_loss,
    "total_loss": total_loss,
    "jaccard_score": jaccard_score,
    "calculate_iou": calculate_iou
})

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

    # Vérification de l'image d'entrée
    print(f"Image shape: {image.shape}")
    #Image.fromarray(image).save("debug_input_image.png")  # Sauvegarde l'image pour débogage


    # Prétraitement : redimensionner l'image à la taille attendue par le modèle
    image_resized = cv2.resize(image, (512, 256))  # Dimensions attendues par le modèle
    image_resized = image_resized / 255.0  # Normaliser les pixels entre 0 et 1
    image_resized = np.expand_dims(image_resized, axis=0)  # Ajouter une dimension pour le batch

    # Prédire le masque
    predicted_mask = model.predict(image_resized)[0]  # Première image du batch
    print(f"Predicted mask shape: {predicted_mask.shape}")
    print(f"Unique values in predicted mask: {np.unique(predicted_mask)}")

    # Post-traitement : convertir les probabilités en classes
    predicted_mask = np.argmax(predicted_mask, axis=-1)

    # Redimensionner le masque à la taille originale de l'image d'entrée
    predicted_mask_resized = cv2.resize(predicted_mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Coloriser le masque
    predicted_mask_colored = colorize_mask(predicted_mask_resized)

    # Sauvegarder le masque coloré dans un fichier temporaire
    mask_image = Image.fromarray(predicted_mask_colored)
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

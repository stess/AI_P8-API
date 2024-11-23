import logging
from flask import Flask, request, jsonify, send_file
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from keras.saving import register_keras_serializable
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import io
import base64
import gdown
import os

# Configurer le logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


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
    score = (2. * intersection + smooth) / \
        (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


@register_keras_serializable()
def dice_loss(y_true, y_pred):
    return 1 - dice_coeff(y_true, y_pred)


@register_keras_serializable()
def total_loss(y_true, y_pred):
    binary_crossentropy_loss = tf.keras.losses.binary_crossentropy(
        y_true, y_pred)
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


def decode_image(base64_str):
    """
    Décoder une image encodée en base64 en un tableau numpy.
    """
    try:
        decoded_bytes = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(decoded_bytes)).convert("RGB")
        logging.info("Image décodée avec succès.")
        return np.array(image)
    except Exception as e:
        logging.error(f"Erreur lors du décodage de l'image : {e}")
        raise


def encode_image(image_array):
    """
    Encoder un tableau numpy en base64.
    """
    try:
        pil_image = Image.fromarray(image_array)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.read()).decode("utf-8")
        logging.info("Image encodée avec succès en base64.")
        return encoded_image
    except Exception as e:
        logging.error(f"Erreur lors de l'encodage de l'image : {e}")
        raise


@app.route("/predict", methods=["POST"])
def predict_mask():
    logging.info("Requête reçue pour la prédiction.")

    # Vérifier la présence de l'image dans la requête
    data = request.json
    if "image" not in data:
        logging.error("Aucune image trouvée dans la requête.")
        return jsonify({"error": "Image manquante dans la requête"}), 400

    # Décoder l'image encodée en base64
    try:
        input_image = decode_image(data["image"])
        logging.info(f"Dimensions de l'image décodée : {input_image.shape}")
    except Exception as e:
        return jsonify({"error": f"Impossible de décoder l'image : {e}"}), 400

    # Prétraiter l'image
    try:
        image_resized = cv2.resize(input_image, (512, 256))  # Redimensionner
        image_resized = image_resized / 255.0  # Normaliser
        # Ajouter une dimension pour le batch
        image_resized = np.expand_dims(image_resized, axis=0)
        logging.info("Image prétraitée pour la prédiction.")
    except Exception as e:
        logging.error(f"Erreur lors du prétraitement de l'image : {e}")
        return jsonify({"error": f"Erreur lors du prétraitement : {e}"}), 500

    # Prédire le masque
    try:
        predicted_mask = model.predict(image_resized)[0]
        logging.info(
            f"Prédiction réussie. Dimensions du masque : {predicted_mask.shape}")
    except Exception as e:
        logging.error(f"Erreur lors de la prédiction : {e}")
        return jsonify({"error": f"Erreur lors de la prédiction : {e}"}), 500

    # Post-traitement : convertir et redimensionner le masque
    try:
        predicted_mask = np.argmax(predicted_mask, axis=-1)
        predicted_mask_resized = cv2.resize(
            predicted_mask.astype(np.uint8),
            (input_image.shape[1], input_image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        predicted_mask_colored = colorize_mask(predicted_mask_resized)
        logging.info("Masque prédictif colorisé avec succès.")
    except Exception as e:
        logging.error(f"Erreur lors du post-traitement : {e}")
        return jsonify({"error": f"Erreur lors du post-traitement : {e}"}), 500

    # Encoder le masque en base64
    try:
        mask_encoded = encode_image(predicted_mask_colored)
        logging.info("Masque prédictif encodé avec succès.")
    except Exception as e:
        logging.error(f"Erreur lors de l'encodage du masque : {e}")
        return jsonify({"error": f"Erreur lors de l'encodage : {e}"}), 500

    # Retourner l'image encodée en base64
    return jsonify({"predicted_image": mask_encoded})


@app.route("/")
def home():
    return "API de prédiction de masque est opérationnelle."


# Lancer l'application
if __name__ == "__main__":
    logging.info("Démarrage de l'application Flask...")
    app.run(host="0.0.0.0", port=5000)
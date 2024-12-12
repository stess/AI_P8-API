from flask import Flask, request, jsonify
#from tensorflow.keras.models import load_model
#import tensorflow.keras.backend as K
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
        return np.array(image)
    except Exception as e:
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
        return encoded_image
    except Exception as e:
        raise


@app.route("/predict", methods=["POST"])
def predict_mask():

    # Vérifier la présence de l'image dans la requête
    data = request.json
    if "image" not in data:
        return jsonify({"error": "Image manquante dans la requête"}), 400

    # Décoder l'image encodée en base64
    try:
        input_image = decode_image(data["image"])
    except Exception as e:
        return jsonify({"error": f"Impossible de décoder l'image : {e}"}), 400

    # Prétraiter l'image
    try:
        image_resized = cv2.resize(input_image, (512, 256))  # Redimensionner
        image_resized = image_resized / 255.0  # Normaliser
        # Ajouter une dimension pour le batch
        image_resized = np.expand_dims(image_resized, axis=0)
    except Exception as e:
        return jsonify({"error": f"Erreur lors du prétraitement : {e}"}), 500

    # Prédire le masque
    try:
        predicted_mask = model.predict(image_resized)[0]
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la prédiction : {e}"}), 500

    # Post-traitement : convertir et redimensionner le masque
    try:
        predicted_mask = np.argmax(
            predicted_mask, axis=-1)  # Indices des classes
        predicted_mask_resized = cv2.resize(
            predicted_mask.astype(np.uint8),
            (input_image.shape[1], input_image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        # Appliquer un étirement d'histogramme pour améliorer les contrastes
        min_val = np.min(predicted_mask_resized)
        max_val = np.max(predicted_mask_resized)
        if max_val > min_val:  # S'assurer que l'étirement est possible
            predicted_mask_resized = (
                (predicted_mask_resized - min_val) * (255 / (max_val - min_val))
            ).astype(np.uint8)
    except Exception as e:
        return jsonify({"error": f"Erreur lors du post-traitement : {e}"}), 500

    # Encoder le masque en niveaux de gris
    try:
        # Convertir le masque en image PIL en niveaux de gris
        grayscale_mask = Image.fromarray(predicted_mask_resized, mode="L")
        buffer = io.BytesIO()
        grayscale_mask.save(buffer, format="PNG")
        buffer.seek(0)
        mask_encoded = base64.b64encode(buffer.read()).decode("utf-8")
    except Exception as e:
        return jsonify({"error": f"Erreur lors de l'encodage : {e}"}), 500

    # Retourner l'image encodée en base64
    return jsonify({"predicted_image": mask_encoded})


@app.route("/")
def home():
    return "API de prédiction de masque est opérationnelle."


# Lancer l'application
if __name__ == "__main__":
    app.run()

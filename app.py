from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

# Initialiser l'application Flask
app = Flask(__name__)


@app.route("/")
def home():
    return "API de prédiction de masque est opérationnelle."


# Lancer l'application
if __name__ == "__main__":
    app.run()

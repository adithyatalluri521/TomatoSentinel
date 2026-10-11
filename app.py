
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import tempfile
import base64
from uuid import uuid4

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), "tomatosentinel_uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

classes = [
    "Bacterial Spot",
    "Early Blight",
    "Healthy",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot"
]

model = None


def get_model():
    global model
    if model is None:
        model = load_model(MODEL_PATH, compile=False)
    return model


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload")
def upload():
    return render_template("upload.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")

    if file is None or not file.filename:
        return "Please select a leaf image.", 400

    extension = os.path.splitext(file.filename)[1].lower()
    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    if extension not in allowed_extensions:
        return "Upload a JPG, JPEG, PNG, or WEBP image.", 400

    filepath = os.path.join(UPLOAD_FOLDER, f"{uuid4().hex}{extension}")

    try:
        file.save(filepath)

        with open(filepath, "rb") as uploaded_image:
            image_bytes = uploaded_image.read()

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp"
        }[extension]

        img = image.load_img(filepath, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = get_model().predict(img_array, verbose=0)
        predicted_class = classes[int(np.argmax(prediction))]
        confidence = round(float(np.max(prediction)) * 100, 2)

        image_data = f"data:{mime_type};base64,{image_base64}"

        return render_template(
            "result.html",
            prediction=predicted_class,
            confidence=confidence,
            filename="",
            image_data=image_data
        )

    except Exception:
        app.logger.exception("Disease prediction failed")
        return "Prediction failed. Please try another image.", 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == "__main__":
    app.run(debug=True)

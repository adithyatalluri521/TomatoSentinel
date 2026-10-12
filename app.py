
import os
import uuid
import logging
import numpy as np

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")

UPLOAD_FOLDER = (
    "/tmp/tomato_uploads"
    if os.environ.get("VERCEL")
    else os.path.join(BASE_DIR, "static", "uploads")
)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

classes = ["Early Blight", "Late Blight", "Healthy"]

model = load_model(MODEL_PATH, compile=False)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload")
def upload():
    return render_template("upload.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return "Please select a tomato leaf image.", 400

        file = request.files["file"]

        if not file.filename:
            return "Please select a tomato leaf image.", 400

        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = model.predict(img_array, verbose=0)

        if prediction.shape[-1] != len(classes):
            app.logger.error(
                "Model output count: %s; configured classes: %s",
                prediction.shape[-1],
                len(classes),
            )
            return "Model and class configuration do not match.", 500

        predicted_index = int(np.argmax(prediction[0]))
        predicted_class = classes[predicted_index]
        confidence = round(float(np.max(prediction[0])) * 100, 2)

        image_data = None

        if os.environ.get("VERCEL"):
            import base64

            with open(filepath, "rb") as img_file:
                encoded = base64.b64encode(img_file.read()).decode("utf-8")

            image_data = f"data:image/jpeg;base64,{encoded}"

        return render_template(
            "result.html",
            prediction=predicted_class,
            confidence=confidence,
            filename=filename,
            image_data=image_data,
        )

    except Exception:
        app.logger.exception("Disease prediction failed")
        return "Prediction failed. Please check the application logs.", 500


if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load Model
model = load_model("model.h5")

# Disease Classes
classes = [
    "Bacterial Spot",
    "Early Blight",
    "Healthy",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot"
]

# Upload Folder
UPLOAD_FOLDER = "static/uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# HOME PAGE
@app.route("/")
def home():
    return render_template("index.html")


# UPLOAD PAGE
@app.route("/upload")
def upload():
    return render_template("upload.html")


# PREDICTION PAGE
@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["file"]

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)

    file.save(filepath)

    # Image Processing
    img = image.load_img(filepath, target_size=(128 , 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    prediction = model.predict(img_array)

    predicted_class = classes[np.argmax(prediction)]

    confidence = round(float(np.max(prediction)) * 100, 2)

    # Result Page
    return render_template(
        "result.html",
        prediction=predicted_class,
        confidence=confidence,
        filename=file.filename
    )


# RUN APP
if __name__ == "__main__":
    app.run(debug=True)
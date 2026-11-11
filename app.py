
import os
from uuid import uuid4

import numpy as np
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pneumonia_model.keras")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

# Load the trained model
model = load_model(MODEL_PATH)


def preprocess_image(image_path):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize((224, 224))
        img_array = np.asarray(img, dtype=np.float32)

    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    uploaded_file = request.files.get("file")

    if not uploaded_file or not uploaded_file.filename:
        return render_template(
            "index.html", error="Please select an X-ray image."
        )

    # Use a generated filename instead of trusting the original filename
    extension = os.path.splitext(uploaded_file.filename)[1].lower()
    if extension not in {".png", ".jpg", ".jpeg"}:
        return render_template(
            "index.html", error="Please upload a PNG or JPG image."
        )

    filename = f"{uuid4().hex}{extension}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    uploaded_file.save(filepath)

    try:
        processed_image = preprocess_image(filepath)
        prediction = float(model.predict(processed_image, verbose=0)[0][0])
    except (UnidentifiedImageError, OSError, ValueError):
        if os.path.exists(filepath):
            os.remove(filepath)
        return render_template(
            "index.html", error="The uploaded file is not a valid image."
        )

    if prediction > 0.5:
        result = "PNEUMONIA"
        confidence = prediction * 100
    else:
        result = "NORMAL"
        confidence = (1 - prediction) * 100

    image_path = f"uploads/{filename}"

    return render_template(
        "index.html",
        result=result,
        confidence=f"{confidence:.2f}%",
        image_path=image_path,
    )


@app.errorhandler(413)
def file_too_large(error):
    return render_template(
        "index.html", error="Image size must be 10 MB or less."
    ), 413


if __name__ == "__main__":
    app.run(debug=True)

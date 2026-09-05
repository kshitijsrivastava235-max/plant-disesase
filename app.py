
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import json, os
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

app = Flask(__name__)

try:
    model = tf.keras.models.load_model("models/plant_model.h5")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

try:
    with open("models/class_names.json") as f:
        class_names = json.load(f)
except Exception as e:
    print(f"Error loading class names: {e}")
    class_names = []

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict(img_path):
    if model is None:
        raise Exception("Model not loaded. Please ensure plant_model.h5 exists in models folder.")
    
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array, verbose=0)
    idx = np.argmax(pred)
    confidence = float(np.max(pred))

    label = class_names[idx].replace("___"," ").title()
    return label, confidence

@app.route("/favicon.ico")
def favicon():
    return "", 204  # Return empty response with 204 No Content

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.route("/", methods=["GET","POST"])
def index():
    prediction=None
    confidence=None
    img_path=None
    error=None

    if request.method=="POST":
        try:
            if 'file' not in request.files:
                error = "No file selected"
            else:
                file = request.files['file']
                if file.filename == '':
                    error = "No file selected"
                elif not allowed_file(file.filename):
                    error = "Invalid file type. Please upload PNG, JPG, JPEG, or GIF"
                else:
                    filename = secure_filename(file.filename)
                    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(path)
                    prediction, confidence = predict(path)
                    img_path = '/' + path.replace('\\', '/')
        except Exception as e:
            error = f"Error processing image: {str(e)}"

    return render_template("index.html", prediction=prediction, confidence=confidence, img_path=img_path, error=error)

if __name__ == "__main__":
    app.run(debug=True, port=5000)

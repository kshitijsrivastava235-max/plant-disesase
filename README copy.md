# Plant Disease Detection (Advanced)

A Flask-based web application that uses deep learning (MobileNetV2) to detect plant diseases from leaf images.

## Features
- 🎯 Real-time plant disease detection
- 📁 Drag-and-drop file upload
- 📊 Confidence score display
- 🎨 Beautiful, responsive UI
- ⚡ Fast predictions using MobileNetV2
- 🔒 Secure file handling

## Setup

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Prepare Dataset (Optional - Only if training)
Place your dataset in the following structure:
```
dataset/
├── Pepper__bell___Bacterial_spot/
│   └── *.jpg
├── Pepper__bell___healthy/
│   └── *.jpg
├── Potato___Early_blight/
│   └── *.jpg
├── Potato___healthy/
│   └── *.jpg
├── Potato___Late_blight/
│   └── *.jpg
├── Tomato__Target_Spot/
│   └── *.jpg
├── Tomato__Tomato_mosaic_virus/
│   └── *.jpg
├── Tomato__Tomato_YellowLeaf__Curl_Virus/
│   └── *.jpg
├── Tomato_Bacterial_spot/
│   └── *.jpg
├── Tomato_Early_blight/
│   └── *.jpg
├── Tomato_healthy/
│   └── *.jpg
├── Tomato_Late_blight/
│   └── *.jpg
├── Tomato_Leaf_Mold/
│   └── *.jpg
├── Tomato_Septoria_leaf_spot/
│   └── *.jpg
└── Tomato_Spider_mites_Two_spotted_spider_mite/
    └── *.jpg
```

### Step 3: Train Model (Optional)
If you want to train a new model with your dataset:
```bash
python train.py
```
This will:
- Load and augment images from the dataset folder
- Train MobileNetV2 with 80/20 train/validation split
- Save the trained model to `models/plant_model.h5`
- Save class names to `models/class_names.json`
- Generate accuracy plot in `models/accuracy.png`

### Step 4: Run the Application
```bash
python app.py
```

The application will start at: **http://127.0.0.1:5000**

## Usage

1. Open http://127.0.0.1:5000 in your browser
2. Upload a plant leaf image (PNG, JPG, JPEG, or GIF)
3. Click "🔍 Predict Disease" button
4. View the prediction result with confidence percentage

## API Endpoints

- `GET /` - Main web interface
- `POST /` - Submit image for prediction
- `GET /health` - Health check (returns JSON status)
- `GET /favicon.ico` - Favicon placeholder

## Project Structure

```
plant/
├── app.py              # Flask application
├── train.py            # Model training script
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── models/
│   ├── plant_model.h5     # Trained model
│   └── class_names.json   # Disease class names
├── templates/
│   └── index.html     # Web interface
├── static/
│   ├── css/
│   │   └── style.css  # Styling
│   └── uploads/       # Uploaded images (auto-created)
└── dataset/           # Training dataset (if training)
```

## Model Details

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Input Size**: 224x224 pixels
- **Image Preprocessing**: Normalization to [0, 1]
- **Training**:
  - Optimizer: Adam
  - Loss: Categorical Crossentropy
  - Batch Size: 32
  - Epochs: 10 (with early stopping)
  - Callbacks: EarlyStopping, ModelCheckpoint

## Supported Diseases

The model detects the following plant diseases:
- Pepper Bell - Bacterial Spot
- Pepper Bell - Healthy
- Potato - Early Blight
- Potato - Healthy
- Potato - Late Blight
- Tomato - Target Spot
- Tomato - Tomato Mosaic Virus
- Tomato - Tomato Yellow Leaf Curl Virus
- Tomato - Bacterial Spot
- Tomato - Early Blight
- Tomato - Healthy
- Tomato - Late Blight
- Tomato - Leaf Mold
- Tomato - Septoria Leaf Spot
- Tomato - Spider Mites (Two Spotted)

## Requirements

See `requirements.txt` for all dependencies:
- TensorFlow >= 2.10
- NumPy
- Flask
- Pillow
- OpenCV
- Matplotlib
- Werkzeug

## Troubleshooting

### Model not loading
- Ensure `models/plant_model.h5` exists
- Check that the model file is not corrupted
- Try re-training the model with `python train.py`

### Port 5000 already in use
- Change the port in `app.py`: `app.run(debug=True, port=5001)`
- Or kill the process using port 5000

### Slow predictions
- First prediction may be slow due to TensorFlow initialization
- Subsequent predictions will be faster
- Run on a machine with GPU for faster predictions

### Upload errors
- Check file format (must be PNG, JPG, JPEG, or GIF)
- Ensure file size is under 10MB
- Check that `static/uploads/` folder exists

## Performance Tips

1. Use GPU: Install CUDA and cuDNN for faster predictions
2. Batch predictions: Process multiple images in sequence
3. Caching: Results are computed fresh for each image
4. Image quality: Better quality images produce better predictions

## License

This project is open source and available for educational purposes.

## Author

Created for plant disease detection and classification tasks.

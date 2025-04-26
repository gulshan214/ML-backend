from flask import Flask, request, jsonify
import numpy as np
import uuid
import tensorflow as tf
from werkzeug.utils import secure_filename
import json
import os

#import uuid
i#mport json
import gdown
#import numpy as np
#import tensorflow as tf
#from flask import Flask, request, jsonify
#from werkzeug.utils import secure_filename


app = Flask(__name__)

# Constants
MODEL_PATH = "farmassit-plant-model.keras"
#UPLOAD_FOLDER = "./uploadimages"
GDRIVE_URL = "https://drive.google.com/uc?id=1w1gQJYKLLpi6-vW4wGyBKA0XE85lt-u4"

# Ensure upload directory exists
#os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Download model from Google Drive if not already present
if not os.path.exists(MODEL_PATH):
    print("Model not found. Downloading from Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    print("Model downloaded successfully.")


model = tf.keras.models.load_model("MODEL_PATH")
label = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Background_without_leaves',
 'Blueberry___healthy',
 'Cherry___Powdery_mildew',
 'Cherry___healthy',
 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn___Common_rust',
 'Corn___Northern_Leaf_Blight',
 'Corn___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy'] #label list

with open("plant_disease.json",'r') as file:
    plant_disease = json.load(file)

def extract_features(image):
    image = tf.keras.utils.load_img(image, target_size=(160, 160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.array([feature])
    return feature

def model_predict(image):
    img = extract_features(image)
    prediction = model.predict(img)
    prediction_label = plant_disease[prediction.argmax()]
    return prediction_label

@app.route('/upload/', methods=['POST'])
def upload_image():
    if 'img' not in request.files:
        return jsonify({"error": "No file part"}), 400
    image = request.files['img']
    if image.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    filename = secure_filename(f"temp_{uuid.uuid4().hex}_{image.filename}")
    image.save(f'./uploadimages/{filename}')
    
    prediction = model_predict(f'./uploadimages/{filename}')
    return jsonify({
        'prediction': prediction,
        'imagepath': f'/uploadimages/{filename}'
    })

if __name__ == "__main__":
    app.run(debug=True)

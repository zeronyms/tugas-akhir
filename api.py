from flask import Flask, request, jsonify
from PIL import Image
import tensorflow as tf
import numpy as np
import mediapipe as mp
import io
import base64

app = Flask(__name__)
model = tf.keras.models.load_model("best_model.keras")
IMG_SIZE = (224, 224)

mp_face_detection = mp.solutions.face_detection

def detect_and_crop_face(image_pil):
    image_np = np.array(image_pil)
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(image_np)

        if results.detections:
            bbox = results.detections[0].location_data.relative_bounding_box
            h, w, _ = image_np.shape
            x, y, width, height = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
            face = image_np[y:y+height, x:x+width]
            face = Image.fromarray(face).resize(IMG_SIZE)

            # Convert cropped face to base64
            buffer = io.BytesIO()
            face.save(buffer, format="JPEG")
            face_b64 = base64.b64encode(buffer.getvalue()).decode()

            return face, face_b64
    return None, None

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_pil = Image.open(request.files['image']).convert('RGB')
    face_img, face_b64 = detect_and_crop_face(image_pil)
    if face_img is None:
        return jsonify({'error': 'Face not detected'}), 400

    image_arr = tf.keras.applications.resnet50.preprocess_input(np.expand_dims(np.array(face_img), axis=0))
    pred = model.predict(image_arr)[0][0]
    label = int(pred > 0.5)

    return jsonify({
        'prediction': float(pred),
        'label': label,
        'face_image': face_b64
    })
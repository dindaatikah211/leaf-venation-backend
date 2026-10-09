import io
import logging
import os

from leaf_guard import leaf_score

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.models import load_model

CLASS_NAMES = ['arcuate', 'palmate', 'parallel', 'pinnate']
IMAGE_SIZE = (224, 224)
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def rescale(batch):
    return batch / 255.0


MODELS = {
    'efficientnet': ('model/efficientnet_leaf.h5', efficientnet_preprocess),
    'mobilenet': ('model/mobilenet_leaf.h5', rescale),
}

MODEL_TYPE = os.getenv('MODEL_TYPE', 'efficientnet')
LEAF_THRESHOLD = float(os.getenv('LEAF_THRESHOLD', '0.5'))
ALLOWED_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(',')

if MODEL_TYPE not in MODELS:
    raise ValueError(f"MODEL_TYPE harus salah satu dari: {', '.join(MODELS)}")

model_path, preprocess = MODELS[MODEL_TYPE]
model = load_model(model_path, compile=False)

app = Flask(__name__)
app.logger.setLevel(logging.INFO)
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_BYTES
CORS(app, origins=ALLOWED_ORIGINS)


def load_image(file):
    return Image.open(io.BytesIO(file.read())).convert('RGB')


def to_batch(image):
    resized = image.resize(IMAGE_SIZE, Image.NEAREST)
    batch = np.expand_dims(np.asarray(resized, dtype='float32'), axis=0)
    return preprocess(batch)


@app.get('/health')
def health():
    return jsonify(status='ok', model=MODEL_TYPE)


@app.post('/predict')
def predict():
    file = request.files.get('file')
    if file is None:
        return jsonify(error='File tidak ditemukan'), 400

    try:
        image = load_image(file)
    except (UnidentifiedImageError, OSError):
        return jsonify(error='File bukan gambar yang valid'), 400

    score = leaf_score(image)
    app.logger.info('leaf_score=%.3f', score)
    if score < LEAF_THRESHOLD:
        return jsonify(
            error='Gambar ini sepertinya bukan daun. Coba foto satu daun dengan jelas.',
            leaf_score=score,
        ), 422

    scores = model.predict(to_batch(image), verbose=0)[0]
    best = int(np.argmax(scores))

    return jsonify(
        label=CLASS_NAMES[best],
        confidence=float(scores[best]),
        probabilities={name: float(value) for name, value in zip(CLASS_NAMES, scores)},
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify(error='Ukuran file maksimal 5 MB'), 413


if __name__ == '__main__':
    app.run(port=5000)
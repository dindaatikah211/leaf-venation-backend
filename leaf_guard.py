import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

IMAGE_SIZE = (224, 224)

model = load_model('model/leaf_guard.keras', compile=False)


def leaf_score(image: Image.Image) -> float:
    resized = image.convert('RGB').resize(IMAGE_SIZE, Image.BILINEAR)
    batch = np.expand_dims(np.asarray(resized, dtype='float32'), axis=0)
    return float(model.predict(batch, verbose=0)[0][0])
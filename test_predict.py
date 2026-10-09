import sys

import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

CLASS_NAMES = ['arcuate', 'palmate', 'parallel', 'pinnate']

model = load_model('model/efficientnet_leaf.h5', compile=False)
path = sys.argv[1]

keras_image = image.load_img(path, target_size=(224, 224))
keras_batch = preprocess_input(np.expand_dims(image.img_to_array(keras_image), axis=0))

pil_image = Image.open(path).convert('RGB').resize((224, 224), Image.NEAREST)
pil_batch = preprocess_input(np.expand_dims(np.asarray(pil_image, dtype='float32'), axis=0))

print('selisih piksel maksimum:', float(np.abs(keras_batch - pil_batch).max()))

for name, batch in (('keras', keras_batch), ('pil', pil_batch)):
    scores = model.predict(batch, verbose=0)[0]
    print(name, {label: round(float(score), 4) for label, score in zip(CLASS_NAMES, scores)})
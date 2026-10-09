import os

os.environ.setdefault('USE_TF', '0')

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = 'openai/clip-vit-base-patch32'

LEAF_PROMPTS = [
    'a close-up photo of a single leaf',
    'a close-up photo of a leaf with visible veins',
    'a macro photo of a green leaf',
    'a close-up photo of leaves on a branch',
    'a close-up photo of rice or grass leaf blades',
]

OTHER_PROMPTS = [
    'a landscape photo of hills and sky',
    'a photo of a grass field or meadow',
    'a photo of a garden or a forest from far away',
    'a photo of a tree',
    'a photo of a flower',
    'a photo of a person',
    'a photo of an animal',
    'a photo of a building or a room',
    'a photo of an object on a table',
    'a photo of food',
    'a drawing or an illustration',
    'a digital artwork',
    'a poster, collage, or screenshot',
]

PROMPTS = LEAF_PROMPTS + OTHER_PROMPTS

processor = CLIPProcessor.from_pretrained(MODEL_NAME)
clip = CLIPModel.from_pretrained(MODEL_NAME).eval()


def prompt_probabilities(image: Image.Image) -> torch.Tensor:
    inputs = processor(text=PROMPTS, images=image, return_tensors='pt', padding=True)
    with torch.no_grad():
        return clip(**inputs).logits_per_image.softmax(dim=1)[0]


def leaf_score(image: Image.Image) -> float:
    return float(prompt_probabilities(image)[: len(LEAF_PROMPTS)].sum())
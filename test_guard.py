import sys

from PIL import Image

from leaf_guard import LEAF_PROMPTS, PROMPTS, prompt_probabilities

for path in sys.argv[1:]:
    probabilities = prompt_probabilities(Image.open(path).convert('RGB'))
    score = float(probabilities[: len(LEAF_PROMPTS)].sum())
    top = probabilities.topk(3)
    print(path)
    print(f'  leaf_score={score:.3f}')
    for value, index in zip(top.values, top.indices):
        print(f'  {float(value):.3f}  {PROMPTS[int(index)]}')
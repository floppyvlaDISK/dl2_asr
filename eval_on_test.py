import torch
import evaluate
import sys

from datasets import load_dataset
from transformers import pipeline
from tqdm import tqdm
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

if len(sys.argv) != 2:
    print("Usage: python fine_tune.py <models/model_name>")
    sys.exit(1)


ds = load_dataset("./spread_flat_dataset.py",
                  "chunk_100",
                  trust_remote_code=True,
                  split="test")

model_name = sys.argv[1]
pipe = pipeline(task="automatic-speech-recognition", model=model_name, tokenizer=model_name)

normalizer = BasicTextNormalizer()
total_examples = len(ds)
total_correct = 0

for i in tqdm(range(total_examples)):
    example = ds[i]
    expected = normalizer(example['transcript'])
    actual = normalizer(pipe(example['audio'])['text'])
    if actual == expected:
        total_correct += 1

accuracy = total_correct / total_examples
print("Accuracy:", accuracy)

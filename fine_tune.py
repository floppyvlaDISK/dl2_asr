from datasets import load_dataset, DatasetDict
from transformers import WhisperFeatureExtractor

common_voice = load_dataset("./flat_dataset.py", trust_remote_code=True)
print(common_voice)

ds=common_voice["train"]
print(ds[0])
audio_files = ds['audio']
transcripts = ds['transcript']
for i in range(5):
    print(transcripts[i])
for i in range(5):
    print(audio_files[i])


#feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

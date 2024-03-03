import torch
import evaluate
import sys

from datasets import load_dataset, DatasetDict
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from functools import partial

if len(sys.argv) != 2:
    print("Usage: python fine_tune.py <model_version>")
    sys.exit(1)

_MODEL_NAME = "openai/whisper-small"
_MODEL_TYPE = "small"
_DS_PERCENTILE = "0_5p"

ds = DatasetDict()
ds["train"] = load_dataset("./flat_dataset.py",
                           trust_remote_code=True,
                           split="train[:5%]")
ds["validation"] = load_dataset("./flat_dataset.py",
                                trust_remote_code=True,
                                split="validation[:5%]")

processor = WhisperProcessor.from_pretrained(_MODEL_NAME,
                                             language="ukrainian",
                                             task="transcribe")

def prepare_dataset(example):
    audio = example["audio"]

    example = processor(audio=audio["array"],
                        sampling_rate=audio["sampling_rate"],
                        text=example["transcript"])

    # compute input length of audio sample in seconds
    example["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    return example

ds = ds.map(prepare_dataset,
            remove_columns=ds.column_names["train"],
            load_from_cache_file=True,
            cache_file_names={
            "train": f"datasets/flat_map_caches/{_MODEL_TYPE}/{_DS_PERCENTILE}/train.arrow",
            "validation": f"datasets/flat_map_caches/{_MODEL_TYPE}/{_DS_PERCENTILE}/dev.arrow"
            },
            num_proc=1)


def is_audio_in_length_range(length):
    return length < 30.0

ds["train"] = ds["train"].filter(is_audio_in_length_range,
                                 input_columns=["input_length"],
                                 load_from_cache_file=True,
                                 cache_file_name=f"datasets/flat_filter_caches/{_MODEL_TYPE}/{_DS_PERCENTILE}/train.arrow")

ds["validation"] = ds["validation"].filter(is_audio_in_length_range,
                                           input_columns=["input_length"],
                                           load_from_cache_file=True,
                                           cache_file_name=f"datasets/flat_filter_caches/{_MODEL_TYPE}/{_DS_PERCENTILE}/dev.arrow")


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [
            {"input_ids": feature["labels"]} for feature in features
        ]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

metric = evaluate.load("wer")

normalizer = BasicTextNormalizer()

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # compute orthographic wer
    wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)

    # compute normalised WER
    pred_str_norm = [normalizer(pred) for pred in pred_str]
    label_str_norm = [normalizer(label) for label in label_str]
    # filtering step to only evaluate the samples that correspond to non-zero references:
    pred_str_norm = [
        pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0
    ]
    label_str_norm = [
        label_str_norm[i]
        for i in range(len(label_str_norm))
        if len(label_str_norm[i]) > 0
    ]

    wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)

    return {"wer_ortho": wer_ortho, "wer": wer}

model = WhisperForConditionalGeneration.from_pretrained(_MODEL_NAME)

# TODO: try removing these
#model.config.forced_decoder_ids = None
#model.config.suppress_tokens = []

# disable cache during training since it's incompatible with gradient checkpointing
model.config.use_cache = False

# set language and task for generation and re-enable cache
model.generate = partial(model.generate,
                         language="ukrainian",
                         task="transcribe",
                         use_cache=True)

training_args = Seq2SeqTrainingArguments(
    output_dir=f"models/{_MODEL_TYPE}_{sys.argv[1]}",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=50,
    max_steps=500,
    gradient_checkpointing=True,
    fp16=True,
    fp16_full_eval=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)
trainer.train()

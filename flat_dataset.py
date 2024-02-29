import os
import datasets
import soundfile as sf

_CITATION = ""
_DESCRIPTION = "Collection of different ua speech datasets"
_HOMEPAGE = ""
_LICENSE = ""
_DATA_DIR = "datasets/flat_dataset"

class FlatDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.1.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "audio": datasets.Audio(sampling_rate=16_000),
                    "transcript": datasets.Value("string"),
                }
            ),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(_DATA_DIR, "train.tsv"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(_DATA_DIR, "dev.tsv"),
                    "split": "dev",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(_DATA_DIR, "test.tsv"),
                    "split": "test"
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        with open(filepath, encoding="utf-8") as f:
            for key, row in enumerate(f):
                row_split = row.split('\t')
                audio = row_split[0]
                transcript = ' '.join(row_split[1:])
                yield key, {
                    "audio": os.path.join(_DATA_DIR, split, audio),
                    "transcript": transcript.strip('\n')
                }

import os
import datasets
import soundfile as sf


_CITATION = ""
_DESCRIPTION = "Collection of different ua speech datasets split into batches for sanity"
_HOMEPAGE = ""
_LICENSE = ""
_DATA_DIR = "datasets/spread_flat_dataset"
_ALL_BATCHES = values = [f"chunk_{i+1}" for i in range(200)]


class SpreadFlatConfig(datasets.BuilderConfig):
    def __init__(self, name, description, citation, homepage):
        super(SpreadFlatConfig, self).__init__(
            name=self.name,
            version=datasets.Version("2.0.0", ""),
            description=self.description,
        )
        self.name = name
        self.description = description
        self.citation = citation
        self.homepage = homepage


def _build_config(name):
    return SpreadFlatConfig(
        name=name,
        description=_DESCRIPTION,
        citation=_CITATION,
        homepage=_HOMEPAGE_URL,
    )


class SpreadFlatDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [_build_config(name) for name in _ALL_BATCHES]

    def _info(self):
        return datasets.DatasetInfo(
            description=self.config.description
            features=datasets.Features(
                {
                    "audio": datasets.Audio(sampling_rate=16_000),
                    "transcript": datasets.Value("string"),
                }
            ),
            homepage=self.config.homepage,
            license=_LICENSE,
            citation=self.config.citation,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(_DATA_DIR, self.config.name, "train.tsv"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(_DATA_DIR, self.config.name, "dev.tsv"),
                    "split": "dev",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(_DATA_DIR, self.config.name, "test.tsv"),
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
                    "audio": os.path.join(_DATA_DIR, self.config.name, split, audio),
                    "transcript": transcript.strip('\n')
                }

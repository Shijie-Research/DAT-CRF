import logging
import os
import shutil
from collections import defaultdict

import datasets
from tqdm import tqdm
from transformers import AutoTokenizer

from awesome_fairseq import ConfigsDict

from . import Dataset

logger = logging.getLogger(__name__)


lang2model = defaultdict(lambda: "bert-base-multilingual-cased")
lang2model.update(
    {
        "en": "roberta-large",
        "zh": "bert-base-chinese",
        "tr": "dbmdz/bert-base-turkish-cased",
        "en-sci": "allenai/scibert_scivocab_uncased",
    },
)


def normalize_text(text):
    return text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"').replace("—", "-")


@Dataset.register("iwslt17_en_zh")
class TEDIWSLT2017(Dataset):
    NAME = "fairseq_iwslt2017"
    HF_PATH = "iwslt2017"
    TOKENIZER = "bert"
    TOKENIZER_CONFIGS = {}
    BPE = "none"
    BPE_CONFIGS = {}

    @classmethod
    def _load(cls, task: str, distilled=False, sep=False, **kwargs):
        lang1, lang2 = task.split("_")[-2:]
        if lang2 == "en":
            lang1, lang2 = lang2, lang1

        from datasets.config import HF_DATASETS_CACHE

        task_dir = os.path.join(HF_DATASETS_CACHE, cls.NAME, task)
        os.makedirs(task_dir, exist_ok=True)

        binarized = os.path.join(task_dir, "binarized" + ("-sep" if sep else "-joined"))
        if not os.path.exists(binarized):
            data_dir = os.path.join(task_dir, "tokenized")
            os.makedirs(data_dir, exist_ok=True)

            data = datasets.load_dataset(
                cls.HF_PATH,
                pair=f"{lang1}-{lang2}",
                is_multilingual=False,
                trust_remote_code=True,
            )

            if sep:
                src_tokenizer = AutoTokenizer.from_pretrained(lang2model[lang1])
                tgt_tokenizer = AutoTokenizer.from_pretrained(lang2model[lang2])
                cls.TOKENIZER_CONFIGS.update({"bert_model_name": tgt_tokenizer.name_or_path})
            else:
                tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
                cls.TOKENIZER_CONFIGS.update({"bert_model_name": tokenizer.name_or_path})

            for split in ["train", "validation", "test"]:
                name = "valid" if split == "validation" else split

                if sep:
                    # with open(os.path.join(data_dir, f"{name}.{lang1}"), "w") as f:
                    #     for text in tqdm(data[split]["translation"]):
                    #         text = text[lang1].lower()
                    #         if lang1 == "zh":
                    #             text = normalize_text(text)
                    #         f.write(" ".join(src_tokenizer.tokenize(text)) + "\n")

                    with open(os.path.join(data_dir, f"{name}.{lang2}"), "w") as f:
                        for text in tqdm(data[split]["translation"]):
                            text = text[lang2].lower()
                            if lang2 == "zh":
                                text = normalize_text(text)
                            f.write(" ".join(tgt_tokenizer.tokenize(text)) + "\n")
                else:
                    with open(os.path.join(data_dir, f"{name}.{lang1}"), "w") as f:
                        for text in tqdm(data[split]["translation"]):
                            text = text[lang1].lower()
                            if lang1 == "zh":
                                text = normalize_text(text)
                            f.write(" ".join(tokenizer.tokenize(text)) + "\n")

                    with open(os.path.join(data_dir, f"{name}.{lang2}"), "w") as f:
                        for text in tqdm(data[split]["translation"]):
                            text = text[lang2].lower()
                            if lang2 == "zh":
                                text = normalize_text(text)
                            f.write(" ".join(tokenizer.tokenize(text)) + "\n")

            preprocess_args = ConfigsDict()

            preprocess_args.update(
                {
                    "--source-lang": lang1,
                    "--target-lang": lang2,
                    "--trainpref": os.path.join(data_dir, "train"),
                    "--validpref": os.path.join(data_dir, "valid"),
                    "--testpref": os.path.join(data_dir, "test"),
                    "--destdir": binarized,
                    "--workers": str(min(8, os.cpu_count())),
                    "--joined-dictionary": binarized.endswith("-joined"),
                },
            )

            from fairseq_cli import preprocess

            preprocess.cli_main(preprocess_args.convert_to_list())

            shutil.rmtree(data_dir)

        return binarized

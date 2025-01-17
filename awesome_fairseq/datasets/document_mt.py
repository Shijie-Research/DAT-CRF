import logging
import os
import shutil

from awesome_fairseq import ConfigsDict

from . import Dataset

logger = logging.getLogger(__name__)


@Dataset.register("iwslt17_en_de", "nc2016_en_de", "europarl7_en_de")
class DocumentMT(Dataset):
    NAME = "fairseq_document_deen"
    HF_PATH = "shijli/fairseq_document_deen"
    TOKENIZER = "moses"
    TOKENIZER_CONFIGS = {}
    BPE = "subword_nmt"
    BPE_CONFIGS = {}

    @classmethod
    def _load(cls, task: str, distilled=False, sep=False, doc=False, **kwargs):
        src_lang, tgt_lang = task.split("_")[-2:]
        cls.TOKENIZER_CONFIGS = {"source_lang": src_lang, "target_lang": tgt_lang}

        task += "-doc" if doc else "-sent"

        from datasets.config import HF_DATASETS_CACHE

        task_dir = os.path.join(HF_DATASETS_CACHE, cls.NAME, task)
        os.makedirs(task_dir, exist_ok=True)

        binarized = os.path.join(task_dir, "binarized" + ("-sep" if sep else "-joined"))
        if not os.path.exists(binarized):
            data_dir = cls._get_extracted_dir(task)

            preprocess_args = ConfigsDict()

            preprocess_args.update(
                {
                    "--source-lang": "de",
                    "--target-lang": "en",
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

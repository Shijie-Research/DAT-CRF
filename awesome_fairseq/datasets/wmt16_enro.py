import logging
import os
import shutil

from awesome_fairseq import ConfigsDict

from . import Dataset

logger = logging.getLogger(__name__)


@Dataset.register("wmt16_en_ro", "wmt16_ro_en")
class WMT2016ENRO(Dataset):
    NAME = "fairseq_wmt16_enro"
    HF_PATH = "shijli/fairseq_wmt16_enro"
    TOKENIZER = "moses"
    TOKENIZER_CONFIGS = {}
    BPE = "subword_nmt"
    BPE_CONFIGS = {}

    @classmethod
    def _load(cls, task: str, distilled=False, sep=False, **kwargs):
        src_lang, tgt_lang = task.split("_")[-2:]
        cls.TOKENIZER_CONFIGS = {"source_lang": src_lang, "target_lang": tgt_lang}

        if distilled:
            task = task + "-distilled"
        else:
            task = "wmt16_en_ro"  # name for raw dataset

        from datasets.config import HF_DATASETS_CACHE

        task_dir = os.path.join(HF_DATASETS_CACHE, cls.NAME, task)
        os.makedirs(task_dir, exist_ok=True)

        binarized = os.path.join(task_dir, "binarized" + ("-sep" if sep else "-joined"))
        if not os.path.exists(binarized):
            data_dir = cls._get_extracted_dir(task)

            preprocess_args = ConfigsDict()

            preprocess_args.update(
                {
                    "--source-lang": "en",
                    "--target-lang": "ro",
                    "--trainpref": os.path.join(data_dir, "train"),
                    "--validpref": os.path.join(data_dir, "valid"),
                    "--testpref": os.path.join(data_dir, "test"),
                    "--destdir": binarized,
                    "--workers": str(min(8, os.cpu_count())),
                    "--joined-dictionary": binarized.endswith("-joined"),
                },
            )

            if distilled:
                # distilled dataset share the same vocabulary as the original dataset, so we reuse the dict.
                dict_dir = binarized.replace("-distilled", "")
                dict_dir = dict_dir.replace("wmt16_ro_en", "wmt16_en_ro")  # raw dataset name

                preprocess_args.update(
                    {
                        "--srcdict": os.path.join(dict_dir, "dict.en.txt"),
                        "--tgtdict": os.path.join(dict_dir, "dict.ro.txt"),
                    },
                )

            from fairseq_cli import preprocess

            preprocess.cli_main(preprocess_args.convert_to_list())

            shutil.rmtree(data_dir)

        return binarized

import logging
import os
import shutil

from awesome_fairseq import ConfigsDict

from . import Dataset

logger = logging.getLogger(__name__)


@Dataset.register("wmt14_en_de", "wmt14_de_en")
class WMT2014ENDE(Dataset):
    NAME = "fairseq_wmt14_ende"
    HF_PATH = "shijli/fairseq_wmt14_ende"
    TOKENIZER = "moses"
    BPE = "subword_nmt"

    @classmethod
    def _load(cls, task: str, distilled=False, sep=False, **kwargs):
        if distilled:
            task = task + "-distilled"
        else:
            task = "wmt14_en_de"  # name for raw dataset

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
                    "--target-lang": "de",
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
                dict_dir = dict_dir.replace("wmt14_de_en", "wmt14_en_de")  # raw dataset name

                preprocess_args.update(
                    {
                        "--srcdict": os.path.join(dict_dir, "dict.en.txt"),
                        "--tgtdict": os.path.join(dict_dir, "dict.de.txt"),
                    },
                )

            from fairseq_cli import preprocess

            preprocess.cli_main(preprocess_args.convert_to_list())

            shutil.rmtree(data_dir)

        return binarized

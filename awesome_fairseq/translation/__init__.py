import json
import logging
from functools import partial

import torch

from awesome_fairseq import CONFIGS, HeadLine, MetaClass, Registry
from awesome_fairseq.datasets import Dataset

logger = logging.getLogger(__name__)

register_translation_tasks = partial(Registry.register_model_to_run_types, "train", "generate")


class Translation(MetaClass):
    task_group: str = "translation"

    def __init__(self, *, task, remaining_args, **kwargs):
        self.task_parser.add_argument("--data-distilled", action="store_true", help="use distilled dataset")
        self.task_parser.add_argument("--data-sep", action="store_true", help="do not use shared dictionary")
        task_args, remaining_args = self.task_parser.parse_known_args(remaining_args)

        # task_name is in the form of `dataset_src_tgt-task_arg1-task_arg2`,
        # e.g. iwslt14_de_en or iwslt14_de_en-distilled.
        self.task_args = {k: v for k, v in vars(task_args).items()}
        _, self.source_lang, self.target_lang = task.split("_")

        super().__init__(task=task, remaining_args=remaining_args, **kwargs)

    def save_dir(self, task=None, model=None):
        task_args = list(self.task_args.keys())
        task_args.sort()

        task = "-".join([self.task, *task_args])
        return super().save_dir(task=task)

    @property
    def train_configs(self):
        configs = super().train_configs
        configs.update(
            {
                # tasks
                "--task": "translation",
                "--source-lang": self.source_lang,
                "--target-lang": self.target_lang,
                "--eval-bleu": True,
                "--eval-bleu-print-samples": True,
                # checkpoint
                "--best-checkpoint-metric": "bleu",
                "--maximize-best-checkpoint-metric": True,
                "--keep-best-checkpoints": "10",
                # optimization
                "--update-freq": "1",
            },
        )
        return configs

    @property
    def generate_configs(self):
        configs = {
            # tasks
            "--task": "translation",
            "--source-lang": self.source_lang,
            "--target-lang": self.target_lang,
            # dataset
            "--batch-size": "128",
            "--required-batch-size-multiple": "1",
            # common_eval
            "--path": "{save_dir}/checkpoint_best.pt",
            "--results-path": "{save_dir}",
            # scoring
            "--scoring": "sacrebleu",
        }
        return configs

    def _post_process_configs(self, grouped_configs):
        if self.run_type == "train":
            # max_tokens is the effective max_tokens in a single device,
            # so we adjust update_freq based on device count
            device_count = torch.cuda.device_count()

            logger.warning(f"Found {device_count} cuda device(s), dynamically adapting update frequency...")
            update_freq = CONFIGS.typed_get("--update-freq", dtype=int)
            max_tokens = CONFIGS.typed_get("--max-tokens", dtype=int)
            if update_freq > 1:  # if update_freq > 1, we reduce this first
                update_freq = update_freq // device_count
                CONFIGS.verbose_update({"--update-freq": str(update_freq), "--max-tokens": str(max_tokens)})
            else:
                max_tokens = max_tokens // device_count
                CONFIGS.verbose_update({"--max-tokens": str(max_tokens), "--update-freq": str(update_freq)})

            logger.warning(f"Effective batch size: {update_freq * max_tokens * device_count} tokens.")

            for key in ["--eval-bleu-args"]:
                CONFIGS.verbose_update({key: json.dumps(grouped_configs.pop(key))})

        # the first _positional_args_ should be data for fairseq translation task
        dataset_cls = Dataset.REGISTRY[self.task]

        data_dir = dataset_cls.load(task=self.task, **self.task_args)
        CONFIGS.append_positional_args(data_dir)

        # we get tokenizer and bpe based on dataset_cls setting, but left an entries for changing it
        tokenizer = dataset_cls.TOKENIZER
        if tokenizer is not None:
            logger.warning(HeadLine.header(f"tokenizer = {tokenizer}", level=2))
            if self.run_type == "train":
                # train stage use --eval-bleu-detok and --eval-bleu-detok-args
                CONFIGS.verbose_update(
                    {
                        "--eval-bleu-detok": tokenizer,
                        "--eval-bleu-detok-args": json.dumps(grouped_configs.pop(tokenizer, {})),
                    },
                )
            else:
                CONFIGS.verbose_update({"--tokenizer": tokenizer, **grouped_configs.pop(tokenizer, {})})

        bpe = dataset_cls.BPE
        if bpe is not None:
            logger.warning(HeadLine.header(f"BPE = {bpe}", level=2))
            if self.run_type == "train":
                CONFIGS.verbose_update({"--eval-bleu-remove-bpe": bpe})
            else:
                CONFIGS.verbose_update({"--post-process": bpe})

        super()._post_process_configs(grouped_configs)

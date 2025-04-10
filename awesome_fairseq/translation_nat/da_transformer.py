import os

from awesome_fairseq import CONFIGS
from awesome_fairseq.constants import PLUGINS_DIR

from .transformer_nat import NATransformerIWSLT14, register_nat_tasks

USER_DIR = os.path.join(PLUGINS_DIR, "DATransformer")
LM_PATH = os.path.join(USER_DIR, "DAG-Search", "models")


register_da_transformer = register_nat_tasks(models="da_transformer")


@register_da_transformer("iwslt14_de_en", "iwslt14_en_de")
class DATransformerIWSLT14(NATransformerIWSLT14):
    def _post_process_configs(self, grouped_configs):
        CONFIGS.verbose_update({"--user-dir": USER_DIR})

        if self.run_type == "train":
            CONFIGS.verbose_update(
                {
                    "--filter-max-length": "{}:{}".format(
                        CONFIGS["--max-source-positions"],
                        CONFIGS["--max-target-positions"],
                    ),
                },
            )

            if CONFIGS["--upsample-base"] != "predict":
                CONFIGS.verbose_update({"--length-loss-factor": "0"})

            if CONFIGS["--decode-strategy"] == "beamsearch":
                CONFIGS.verbose_update({"--decode-lm-path": os.path.join(LM_PATH, f"iwslt14-{self.target_lang}.arpa")})

        super()._post_process_configs(grouped_configs)

    @property
    def train_configs(self):
        configs = super().train_configs
        configs.update(
            {
                # tasks
                "--task": "translation_dat_task",
                "--upsample-base": "source_old",
                "--upsample-scale": "8",
                "--filter-ratio": "2",
                "--skip-invalid-size-inputs-valid-test": True,
                # model
                "--arch": "glat_decomposed_link_iwslt_de_en",
                "--links-feature": "feature:position",
                "--max-source-positions": "136",
                "--max-target-positions": "1088",
                "--decode-strategy": "beamsearch",
                "--decode-upsample-scale": "8.0",
                # criterion
                "--criterion": "nat_dag_loss",
                "--max-transition-length": "99999",
                "--glat-p": "0.5:0.1@200k",
                "--glance-strategy": "number-random",
                "--no-force-emit": True,
                "--label-smoothing": "0.0",
                "--clip-norm": "0.1",
                "--grouped-shuffling": True,
                "--decode-max-batchsize": "128",
                "--batch-size-valid": "128",
                "--max-encoder-batch-tokens": "8000",
                "--max-decoder-batch-tokens": "34000",
            },
        )
        return configs


register_dacrf_transformer = register_nat_tasks(models=["dacrf_transformer", "dacrf_transformer_finetune"])


@register_dacrf_transformer("iwslt14_de_en", "iwslt14_en_de")
class DACRFTransformerIWSLT14(NATransformerIWSLT14):
    def _pre_process_configs(self, grouped_configs):
        if grouped_configs.get("--model-overrides.decode_strategy") == "beamsearch":  # beamsearch at inference time
            lm_path = os.path.join(LM_PATH, "{}-{}.arpa".format(self.task.split("_")[0], self.target_lang))
            grouped_configs.verbose_update(
                {
                    "--model-overrides.decode_lm_path": lm_path,
                    "--model-overrides.decode_max_batchsize@int": CONFIGS["--batch-size"],
                    "--model-overrides.max_encoder_batch_tokens@int": "8000",
                    "--model-overrides.max_decoder_batch_tokens@int": "34000",
                },
            )
        super()._pre_process_configs(grouped_configs)

    def _post_process_configs(self, grouped_configs):
        CONFIGS.verbose_update({"--user-dir": USER_DIR})

        for key, value in CONFIGS.items():
            if isinstance(value, str) and "{max_update}" in value:
                CONFIGS.update({key: value.format(max_update=CONFIGS["--max-update"])})

        if CONFIGS.get("--decode-strategy") == "beamsearch":  # beamsearch at training time
            lm_path = os.path.join(LM_PATH, "{}-{}.arpa".format(self.task.split("_")[0], self.target_lang))
            CONFIGS.verbose_update({"--decode-lm-path": lm_path})

        # go to parent
        super()._post_process_configs(grouped_configs)

    @property
    def train_configs(self):
        configs = super().train_configs
        configs.update(
            {
                # task
                "--filter-max-sizes": "256,1024",
                "--filter-ratio": "2",
                "--skip-invalid-size-inputs-valid-test": True,
                # model
                "--arch": "dacrf_transformer_iwslt_de_en",
                "--glance-p": "0.5:0.1@{max_update}",
                "--upsample-scale": "2",
                "--upsample-target": False,
                "--decode-strategy": "lookahead",
            },
        )
        if "finetune" in self.model:
            configs.update(
                {
                    "--crf-finetuning": True,
                    "--finetune-from-model": "{save_dir}/checkpoint_finetune.pt",
                    "--glance-p": None,
                    "--decode-strategy": "lookahead",
                    "--no-strict-model-load": True,
                    "--length-loss-factor": "0.0",
                    "--lr": "5e-4",
                    "--lr-scheduler": "polynomial_decay",
                    "polynomial_decay.warmup_updates": "0",
                    "polynomial_decay.end_learning_rate": "1e-4",
                    "polynomial_decay.total_num_update": "{max_update}",
                    "--max-update": ("50000", "20"),
                    "--validate-interval-updates": ("1000", "10"),
                    "--save-interval-updates": ("1000", "10"),
                    "--crf-lowrank-approx": "64",
                    "--crf-beam-approx": "64",
                    "--crf-decode-beam": "4",
                    "--patience": "5",
                },
            )
        return configs

    def save_dir(self, task=None, model=None):
        save_dir = super().save_dir(task=task, model=model or "dacrf_transformer")
        if "finetune" in self.model:
            save_dir = os.path.join(save_dir, "finetune" + CONFIGS.pop("--finetune-suffix", ""))
        return save_dir

    @property
    def generate_configs(self):
        configs = super().generate_configs
        configs.update(
            {
                "--model-overrides.glance_p": None,
                "--model-overrides.crf_finetuning": "finetune" in self.model,
                "--model-overrides.decode_alpha@float": "1.0",
                "--model-overrides.decode_beta@int": "1",
                "--model-overrides.crf_decode_beam@int": "4",
                # parameters for beamsearch
                "--model-overrides.decode_max_workers@int": "0",
                "--model-overrides.decode_max_beam_per_length@int": "10",
                "--model-overrides.decode_beamsize@int": "100",
                "--model-overrides.decode_top_cand_n@int": "4",
                "--model-overrides.decode_top_p@float": "0.9",
                "--model-overrides.decode_threads_per_worker@int": "4",
                "--model-overrides.decode_gamma@float": "0.1",
                "--model-overrides.decode_no_consecutive_repeated_ngram@int": "0",
                "--model-overrides.decode_no_repeated_ngram@int": "0",
                "--model-overrides.decode_dedup@bool": False,
            },
        )
        return configs


@register_dacrf_transformer("iwslt17_en_de", "nc2016_en_de", "europarl7_en_de")
class DACRFTransformerSent(DACRFTransformerIWSLT14):
    def _post_process_configs(self, grouped_configs):
        if self.task == "europarl7_en_de" and self.run_type == "train":
            if CONFIGS.typed_get("--upsample-scale", dtype=int) == 8:
                # 8192 token cannot be trained with a 48G GPU. So pity.
                CONFIGS.verbose_update({"--max-tokens": "4096", "--update-freq": "4", "--dropout": "0.2"})
            else:
                CONFIGS.verbose_update({"--max-tokens": "16384", "--update-freq": "1", "--dropout": "0.2"})

        # go to parent
        super()._post_process_configs(grouped_configs)

    @property
    def train_configs(self):
        configs = super().train_configs
        configs.update(
            {
                # task
                "--prepend-bos": True,  # already exists in source sentences
                # model
                "--arch": "dacrf_transformer_wmt_en_de",
            },
        )
        return configs

    @property
    def generate_configs(self):
        configs = super().generate_configs
        configs.update(
            {
                # task
                "--prepend-bos": True,  # already exists in source sentences
            },
        )
        return configs


@register_dacrf_transformer("iwslt17_en_zh", "iwslt17_zh_en")
class DACRFTransformerIWSLT17(DACRFTransformerIWSLT14):
    def __init__(self, *, task, remaining_args, **kwargs):
        remaining_args.append("--data-sep")
        super().__init__(task=task, remaining_args=remaining_args, **kwargs)

    @property
    def train_configs(self):
        configs = super().train_configs
        configs.update(
            {
                # model
                "--arch": "dacrf_transformer_wmt_en_de",
                "--share-all-embeddings": False,
                "--share-decoder-input-output-embed": True,
            },
        )
        return configs


@register_dacrf_transformer("wmt16_en_ro", "wmt16_ro_en")
class DACRFTransformerWMT16(DACRFTransformerIWSLT14):
    def _post_process_configs(self, grouped_configs):
        if self.run_type == "train":
            if CONFIGS.typed_get("--upsample-scale", dtype=int) == 8:
                # 8192 token cannot be trained with a 48G GPU. So pity.
                CONFIGS.verbose_update({"--max-tokens": "4096", "--update-freq": "8"})
            else:
                # dataset, 32K batch size assuming only one GPU
                CONFIGS.verbose_update({"--max-tokens": "16384", "--update-freq": "2"})

        # go to parent
        super()._post_process_configs(grouped_configs)

    @property
    def train_configs(self):
        configs = super().train_configs
        configs.update(
            {
                # model
                "--arch": "dacrf_transformer_wmt_en_de",
                "--dropout": "0.3",
            },
        )
        return configs


@register_dacrf_transformer("wmt14_de_en", "wmt14_en_de")
class DACRFTransformerWMT14(DACRFTransformerIWSLT14):
    def _post_process_configs(self, grouped_configs):
        if self.run_type == "train":
            if CONFIGS.typed_get("--upsample-scale", dtype=int) == 8:
                # 8192 token cannot be trained with a 48G GPU. So pity.
                CONFIGS.verbose_update({"--max-tokens": "4096", "--update-freq": "8"})
            else:
                # dataset, 32K batch size assuming only one GPU
                CONFIGS.verbose_update({"--max-tokens": "16384", "--update-freq": "2"})

        # go to parent
        super()._post_process_configs(grouped_configs)

    @property
    def train_configs(self):
        configs = super().train_configs
        configs.update(
            {
                # model
                "--arch": "dacrf_transformer_wmt_en_de",
                "--dropout": "0.1",
            },
        )
        if "finetune" not in self.model:
            configs.update({"--lr": "7e-4"})
        return configs


@register_dacrf_transformer("iwslt17_en_de-doc", "nc2016_en_de-doc", "europarl7_en_de-doc")
class DACRFTransformerDoc(DACRFTransformerIWSLT14):
    def __init__(self, *, task, remaining_args, **kwargs):
        self.task_args.update(doc=True)

        self.task_parser.add_argument("--global", action="store_true", help="use global attention")
        parsed, remaining_args = self.task_parser.parse_known_args(remaining_args)
        self.is_global = getattr(parsed, "global", False)
        super().__init__(task=task.split("-")[0], remaining_args=remaining_args, **kwargs)

    def _post_process_configs(self, grouped_configs):
        CONFIGS.verbose_update({"--max-target-positions": str(512 * CONFIGS.typed_get("--upsample-scale", dtype=int))})

        # go to parent
        super()._post_process_configs(grouped_configs)

    @property
    def train_configs(self):
        configs = super().train_configs
        configs.update(
            {
                "--filter-max-sizes": None,
                "--filter-ratio": None,
                # task
                "--prepend-bos": True,  # already exists in source sentences
                "--arch": "dacrf_transformer_doc_wmt_en_de",
                "--encoder-ctxlayers": "2",
                "--decoder-ctxlayers": "0",  # use local for all decoder layer
                "--cross-ctxlayers": "0",  # use local for all cross layer
                "--no-decoder-local": True if self.is_global else False,
                "--no-cross-local": True if self.is_global else False,
                "--eval-bleu-args.iter_decode_force_max_iter@bool": True,
                "--eval-bleu-args.iter_decode_max_iter@int": "1",
                # dataset
                "--max-tokens": ("4096", "1024"),
                "--update-freq": ("2", "1"),
            },
        )
        if "europarl7" in self.task:
            configs.update(
                {
                    "--max-tokens": "2048",
                    "--update-freq": "8",
                    "--dropout": "0.2",
                },
            )
        return configs

    @property
    def generate_configs(self):
        configs = super().generate_configs
        configs.update(
            {
                # model
                "--iter-decode-force-max-iter": True,
                "--iter-decode-max-iter": "1",
                "--batch-size": "4",
                "--prepend-bos": True,  # already exists in source sentences
            },
        )
        return configs

    def save_dir(self, task=None, model=None):
        model = "dacrf_transformer" + ("-global" if self.is_global else "-local")
        save_dir = super().save_dir(task=task, model=model)
        return save_dir

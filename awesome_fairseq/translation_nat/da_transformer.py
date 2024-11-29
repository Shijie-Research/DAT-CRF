import os

from awesome_fairseq import CONFIGS
from awesome_fairseq.constants import PLUGINS_DIR

from .transformer_nat import NATransformerIWSLT14, register_nat_tasks

USER_DIR = os.path.join(PLUGINS_DIR, "DATransformer")
LM_PATH = os.path.join(USER_DIR, "DAG-Search", "models")


register_da_transformer = register_nat_tasks(models="da_transformer")


@register_da_transformer("iwslt14_de_en", "iwslt14_en_de")
class DATransformerIWSLT14(NATransformerIWSLT14):
    @property
    def train_configs(self):
        configs = super().train_configs
        configs.update(
            {
                "--user-dir": USER_DIR,
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
                "--max-encoder-batch-tokens": "8000",
                "--max-decoder-batch-tokens": "34000",
            },
        )
        return configs

    def _post_process_configs(self, grouped_configs):
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


register_dacrf_transformer = register_nat_tasks(models=["dacrf_transformer", "dacrf_transformer_finetune"])


@register_dacrf_transformer("iwslt14_de_en", "iwslt14_en_de")
class DACRFTransformerIWSLT14(NATransformerIWSLT14):
    def _post_process_configs(self, *args, **kwargs):
        for key, value in CONFIGS.items():
            if isinstance(value, str) and "{max_update}" in value:
                CONFIGS.update({key: value.format(max_update=CONFIGS["--max-update"])})

        # go to parent
        super()._post_process_configs(*args, **kwargs)

    @property
    def train_configs(self):
        configs = super().train_configs
        configs.update(
            {
                "--user-dir": USER_DIR,
                # task
                "--filter-max-sizes": "256,1024",
                "--filter-ratio": "2",
                "--skip-invalid-size-inputs-valid-test": True,
                # model
                "--arch": "dacrf_transformer_iwslt_de_en",
                "--upsample-scale": "2",
                "--upsample-base": "target",
                "--decode-strategy": "viterbi",
            },
        )
        if "finetune" in self.model:
            configs.update(
                {
                    "--finetuning": True,
                    "--finetune-from-model": "{save_dir}/checkpoint_finetune.pt",
                    "--no-strict-model-load": True,
                    "--length-loss-factor": "0.0",
                    "--lr": "5e-4",
                    "--lr-scheduler": "polynomial_decay",
                    "polynomial_decay.--warmup-updates": "0",
                    "polynomial_decay.--end-learning-rate": "1e-5",
                    "polynomial_decay.--power": "1",
                    "polynomial_decay.--total-num-update": "{max_update}",
                    "--crf-lowrank-approx": "64",
                    "--crf-beam-approx": "64",
                    "--crf-decode-beam": "8",
                },
            )
        else:
            configs.update({"--glance-p": "0.5:0.1@{max_update}"})
        return configs

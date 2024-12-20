from . import Translation, register_translation_tasks

register_tasks = register_translation_tasks(models="transformer")


@register_tasks("iwslt14_de_en", "iwslt14_en_de")
class TransformerIWSLT14(Translation):

    @property
    def train_configs(self):
        configs = super().train_configs
        configs.update(
            {
                # model
                "--arch": "transformer_iwslt_de_en",
                "--dropout": "0.3",
                "--share-all-embeddings": True,
                # tasks
                "--eval-bleu-args.beam@int": "5",
                "--eval-bleu-args.lenpen@float": "1",
                # criterion
                "--criterion": "label_smoothed_cross_entropy",
                "--label-smoothing": "0.1",
                "--report-accuracy": True,
                # optimizer
                "--optimizer": "adam",
                "adam.--adam-betas": "0.9,0.98",
                "adam.--adam-eps": "1e-8",
                "adam.--weight-decay": "0.01",
                # lr_scheduler
                "--lr-scheduler": "inverse_sqrt",
                "inverse_sqrt.--warmup-updates": "4000",
                "inverse_sqrt.--warmup-init-lr": "1e-7",
                # dataset, 8K batch size assuming only one GPU
                "--max-tokens": ("8192", "1024"),
                "--update-freq": "1",
                "--validate-interval": "0",  # do not validate at end_of_epoch
                "--validate-interval-updates": ("1000", "10"),
                # optimization
                "--max-update": ("50000", "20"),
                "--clip-norm": "5.0",
                "--lr": "5e-4",
                "--stop-min-lr": "1e-9",
                # checkpoint
                "--no-epoch-checkpoints": True,  # do not save at end_of_epoch
                "--save-interval": "0",  # do not save at end_of_epoch
                "--save-interval-updates": ("1000", "10"),
                "--keep-interval-updates": "5",
                # tokenizer
                "--tokenizer": "moses",
                "moses.source_lang": self.source_lang,
                "moses.target_lang": self.target_lang,
            },
        )
        return configs

    @property
    def generate_configs(self):
        configs = super().generate_configs
        configs.update(
            {
                # generation
                "--beam": "5",
                "--lenpen": "1",
                # tokenizer
                "--tokenizer": "moses",
                "moses.--source-lang": self.source_lang,
                "moses.--target-lang": self.target_lang,
            },
        )
        return configs


@register_tasks("wmt16_en_ro", "wmt16_ro_en")
class TransformerWMT16(TransformerIWSLT14):

    @property
    def train_configs(self):
        configs = super().train_configs
        configs.update(
            {
                # model
                "--arch": "transformer_wmt_en_de",
                "--dropout": "0.3",
                # dataset, 32K batch size assuming only one GPU
                "--max-tokens": ("16384", "1024"),
                # optimization
                "_debug_::--update-freq": ("2", "1"),
            },
        )
        return configs


@register_tasks("wmt14_de_en", "wmt14_en_de")
class TransformerWMT14(TransformerIWSLT14):
    @property
    def train_configs(self):
        configs = super().train_configs
        configs.update(
            {
                # model
                "--arch": "transformer_wmt_en_de",
                "--dropout": "0.1",
                # dataset, 64K batch size assuming only one GPU
                "--max-tokens": ("16384", "1024"),
                # optimization
                "--lr": "7e-4",
                "_debug_::--update-freq": ("4", "1"),
            },
        )
        return configs


@register_tasks("wmt17_en_zh", "wmt17_zh_en")
class TransformerWMT17(TransformerWMT14):
    @property
    def train_configs(self):
        configs = super().train_configs
        configs.update(
            {
                # model
                "--share-all-embeddings": False,
                "--share-decoder-input-output-embed": True,
            },
        )
        return configs


@register_tasks("iwslt17_en_de", "nc2016_en_de", "europarl7_en_de")
class TransformerSent(TransformerIWSLT14):
    @property
    def train_configs(self):
        configs = super().train_configs
        configs.update(
            {
                # model
                "--arch": "transformer_wmt_en_de",
            },
        )
        if "europarl7" in self.task:
            configs.update(
                {
                    "--max-tokens": ("16384", "1024"),
                    "--dropout": "0.2",
                },
            )

        return configs


@register_tasks("iwslt17_en_zh")
class TransformerIWSLT17(TransformerIWSLT14):
    @property
    def train_configs(self):
        configs = super().train_configs
        configs.update(
            {
                # model
                "--arch": "transformer_wmt_en_de",
                "--share-all-embeddings": False,
                "--share-decoder-input-output-embed": True,
            },
        )
        return configs

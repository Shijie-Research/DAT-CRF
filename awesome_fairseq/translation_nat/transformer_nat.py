from . import NATranslation, register_nat_tasks

register_tasks = register_nat_tasks(models="nat_base")


@register_tasks("iwslt14_de_en", "iwslt14_en_de")
class NATransformerIWSLT14(NATranslation):
    @property
    def train_configs(self):
        configs = super().train_configs
        configs.update(
            {
                # model
                "--arch": "nonautoregressive_transformer_iwslt_de_en",
                "--dropout": "0.3",
                "--share-all-embeddings": True,
                "--encoder-learned-pos": True,
                "--decoder-learned-pos": True,
                "--pred-length-offset": True,
                "--length-loss-factor": "0.1",
                "--activation-fn": "gelu",
                "--apply-bert-init": True,
                # tasks
                "--task": "translation_lev",
                "--noise": "full_mask",
                "--eval-bleu-args.iter_decode_max_iter@int": "1",
                "--eval-bleu-args.iter_decode_with_beam@int": "1",
                "--eval-bleu-args::iter_decode_length_format@str": (None, "oracle"),
                # criterion
                "--criterion": "nat_loss",
                "--label-smoothing": "0.1",
                # optimizer
                "--optimizer": "adam",
                "adam.adam_betas": "0.9,0.98",
                "adam.adam_eps": "1e-8",
                "adam.weight_decay": "0.01",
                # lr_scheduler
                "--lr-scheduler": "inverse_sqrt",
                "inverse_sqrt.warmup_updates": "10000",
                "inverse_sqrt.warmup_init_lr": "1e-7",
                # dataset
                "--max-tokens": ("8192", "1024"),
                "--update-freq": "1",
                "--validate-interval": "0",  # do not validate at end_of_epoch
                "--validate-interval-updates": ("1000", "10"),
                # optimization
                "--max-update": ("200000", "20"),
                "--clip-norm": "10.0",
                "--lr": "5e-4",
                "--stop-min-lr": "1e-9",
                # checkpoint
                "--no-epoch-checkpoints": True,  # do not save at end_of_epoch
                "--save-interval": "0",  # do not save at end_of_epoch
                "--save-interval-updates": ("1000", "10"),
                "--keep-interval-updates": "5",
                "--patience": "20",
            },
        )
        return configs

    @property
    def generate_configs(self):
        configs = super().generate_configs
        configs.update(
            {
                # NAT configs
                "--task": "translation_lev",
                "--iter-decode-max-iter": "1",
                "--iter-decode-with-beam": "1",
            },
        )
        return configs


@register_tasks("wmt16_en_ro", "wmt16_ro_en")
class NATransformerWMT16(NATransformerIWSLT14):
    @property
    def train_configs(self):
        configs = super().train_configs
        configs.update(
            {
                # model
                "--arch": "nonautoregressive_transformer_wmt_en_de",
                "--dropout": "0.3",
                # dataset, 32K batch size assuming only one GPU
                "_debug_::--max-tokens": ("16384", "1024"),
                # optimization
                "_debug_::--update-freq": ("2", "1"),
            },
        )
        return configs


@register_tasks("wmt14_de_en", "wmt14_en_de")
class NATransformerWMT14(NATransformerIWSLT14):
    @property
    def train_configs(self):
        configs = super().train_configs
        configs.update(
            {
                # model
                "--arch": "nonautoregressive_transformer_wmt_en_de",
                "--dropout": "0.1",
                # dataset, 64K batch size assuming only one GPU
                "_debug_::--max-tokens": ("16384", "1024"),
                # optimization
                "--lr": "7e-4",
                "_debug_::--update-freq": ("4", "1"),
            },
        )
        return configs

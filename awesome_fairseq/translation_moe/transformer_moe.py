from . import TranslationMOE, register_translation_moe_tasks

register_tasks = register_translation_moe_tasks(
    models=[
        "transformer_hMoElp",
        "transformer_hMoEup",
        "transformer_sMoElp",
        "transformer_sMoEup",
    ],
)


@register_tasks("iwslt14_de_en", "iwslt14_en_de")
class TransformerMOEIWSLT14(TranslationMOE):
    def __init__(self, *, model, **kwargs):
        self.moe_method = model.split("_")[1]
        super().__init__(model=model, **kwargs)

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
                "--task": "translation_moe",
                "--method": self.moe_method,
                "--mean-pool-gating-network": True,
                "--num-experts": "3",
                "--eval-bleu-args.beam@int": "5",
                "--eval-bleu-args.lenpen@float": "1",
                # criterion
                "--criterion": "label_smoothed_cross_entropy",
                "--label-smoothing": "0.0",
                "--report-accuracy": True,
                # optimizer
                "--optimizer": "adam",
                "adam.adam_betas": "0.9,0.98",
                "adam.adam_eps": "1e-8",
                "adam.weight_decay": "0.01",
                # lr_scheduler
                "--lr-scheduler": "inverse_sqrt",
                "inverse_sqrt.warmup_updates": "4000",
                "inverse_sqrt.warmup_init_lr": "1e-7",
                # dataset, 8K batch size assuming only one GPU
                "--max-tokens": ("8192", "1024"),
                "--update-freq": "1",
                "--validate-interval": "0",  # do not validate at end_of_epoch
                "--validate-interval-updates": ("1000", "10"),
                # optimization
                "--max-update": ("30000", "20"),
                "--clip-norm": "0.0",
                "--lr": "5e-4",
                "--stop-min-lr": "1e-9",
                # checkpoint
                "--no-epoch-checkpoints": True,  # do not save at end_of_epoch
                "--save-interval": "0",  # do not save at end_of_epoch
                "--save-interval-updates": ("1000", "10"),
                "--keep-interval-updates": "5",
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
            },
        )
        return configs

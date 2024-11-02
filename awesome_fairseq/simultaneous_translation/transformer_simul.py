from . import SimulTranslation, register_simul_translation_tasks

register_tasks = register_simul_translation_tasks(
    models=[
        "transformer_hard_aligned",
        "transformer_infinite_lookback",
        "transformer_waitk",
        "transformer_chunkwise",
        "transformer_hard_aligned_fixed_pre_decision",
        "transformer_infinite_lookback_fixed_pre_decision",
        "transformer_waitk_fixed_pre_decision",
    ],
)


@register_tasks("iwslt14_de_en", "iwslt14_en_de")
class SimulTranslationIWSLT14(SimulTranslation):
    def __init__(self, *, model, **kwargs):
        self.simul_method = model.split("_", 1)[1]
        super().__init__(model=model, **kwargs)

    @property
    def train_configs(self):
        configs = super().train_configs
        configs.update(
            {
                # model
                "--arch": "transformer_monotonic_iwslt_de_en",
                "--dropout": "0.3",
                "--share-all-embeddings": True,
                # tasks
                "--eval-bleu-args.beam@int": "5",
                "--eval-bleu-args.lenpen@float": "1",
                # criterion
                "--criterion": "latency_augmented_label_smoothed_cross_entropy",
                "--label-smoothing": "0.1",
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
        configs.update(
            {
                "--simul-type": self.simul_method,
                **self.simul_type_configs,
            },
        )
        return configs

    @property
    def simul_type_configs(self):
        configs = {
            "--no-mass-preservation": False,
            "--attention-eps": None,
            "--noise-var": None,
            "--noise-mean": None,
            "--noise-type": None,
            "--energy-bias": None,
            "--energy-bias-init": None,
        }

        if "hard_aligned" in self.simul_method:
            configs.update({"--latency-weight-var": "0.1"})

        if "infinite_lookback" in self.simul_method:
            configs.update({"--latency-weight-avg": "0.1"})

        if "waitk" in self.simul_method:
            configs.update({"--waitk-lagging": "3"})

        if "fixed_pre_decision" in self.simul_method:
            configs.update(
                {
                    "--fixed-pre-decision-ratio": None,
                    "--fixed-pre-decision-type": None,
                    "--fixed-pre-decision-pad-threshold": None,
                },
            )

        return configs

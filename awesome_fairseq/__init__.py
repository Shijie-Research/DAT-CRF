import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from contextlib import contextmanager
from itertools import product
from pathlib import Path

from .constants import DEBUG_MODE, EXPERIMENT_DIR, WANDB_DISABLED

logger = logging.getLogger(__name__)


class HeadLine:
    default_width = 20

    @classmethod
    def header(cls, msg, level):
        """Formats the message with dashes based on the level and message length."""
        total_width = max(cls.default_width * level, len(msg) + 4)  # longest case is - msg -
        side_width = (total_width - len(msg) - 2) // 2
        return "-" * side_width + f" {msg} " + "-" * side_width

    @classmethod
    def tail(cls, header):
        return "*" * len(header)

    @classmethod
    @contextmanager
    def wrap(cls, msg, level=1, warning=False):
        _header = cls.header(msg, level=level)
        printer = getattr(logger, "warning" if warning else "info")
        printer(_header)
        yield
        _tail = cls.tail(_header)
        printer(_tail)


class Registry:
    """Registry for all tasks"""

    _command_registry = {}
    _model_task_registry = {}
    _class_names = set()

    @classmethod
    def register_run_type(cls, run_type):

        def wrapper(func):
            if run_type in cls._command_registry:
                raise ValueError(f"Cannot register duplicate run type: {run_type}")
            cls._command_registry[run_type] = func
            cls._model_task_registry[run_type] = defaultdict(dict)
            return func

        return wrapper

    @classmethod
    def register_model_to_run_types(cls, *run_types, models):
        if isinstance(models, str):
            models = [models]

        def register_tasks(*tasks):

            def wrapper(task_cls):

                if not issubclass(task_cls, MetaClass):
                    raise ValueError(f"Task class {task_cls.__name__} must extend MetaClass")

                if task_cls.__name__ in cls._class_names:
                    raise ValueError(f"Cannot register {task_cls.__name__} with duplicate class name.")
                cls._class_names.add(task_cls.__name__)

                for run_type in run_types:
                    if run_type not in cls._model_task_registry:
                        raise ValueError(f"Cannot register model to a non-existent run type: {run_type}")

                    for model, task in product(models, tasks):
                        if task in cls._model_task_registry[run_type][model]:
                            raise ValueError(f"Cannot register duplicate task ({task}) to model ({model})")
                        cls._model_task_registry[run_type][model].update({task: task_cls})

                return task_cls

            return wrapper

        return register_tasks

    @classmethod
    def get_run_main(cls, run_type):
        return cls._command_registry[run_type]


class ConfigsDict(dict):
    _positional_args = []

    def verbose_update(self, new_dict):
        for k, v in new_dict.items():
            if k not in self:
                logger.info(f"Add config: {k} = {v}")
            elif v != self[k]:
                logger.info(f"Update config: {k} = {self[k]} -> {v}")
            elif v == self[k]:
                logger.info(f"Unchanged config: {k} = {v}")

        self.update(new_dict)

    def update_from_list(self, args_list, verbose=False):
        configs = {}
        for arg in args_list:
            assert "=" in arg, "Invalid argument format. Please specify configurations in the format `key=value`."

            key, val = arg.split("=", 1)

            if val in ["True", "False", "None"]:
                configs[key] = eval(val)
            else:
                configs[key] = val

        if verbose:
            self.verbose_update(configs)
        else:
            self.update(configs)

    def convert_to_list(self):
        args_list, positional = [], []
        for key, val in self.items():
            if val in [False, None]:
                continue

            if val in [True]:
                args_list += [key]
                continue

            args_list += [key, val]

        return self._positional_args + args_list

    def append_positional_args(self, val):
        self._positional_args.append(val)

    def typed_get(self, key, dtype=str):
        return dtype(self[key])


CONFIGS = ConfigsDict()


class MetaClass:
    """MetaClass for all tasks"""

    task_group: str
    task_parser: argparse.ArgumentParser

    def __init__(self, *, run_type, model, task, remaining_args):
        self.task_parser.add_argument("--save-dir-suffix", type=str, help="suffix for the save directory")
        self.task_parser.add_argument("-h", "--help", action="store_true", help="Show this help message and exit")
        kwargs, remaining_args = self.task_parser.parse_known_args(remaining_args)

        self.run_type = run_type
        self.model = model
        self.task = task
        self.kwargs = vars(kwargs)

        if self.kwargs.get("help", False):
            self.task_parser.print_help()
            sys.exit(0)

        # setup wand project name
        if self.run_type == "train" and not DEBUG_MODE:
            os.environ["WANDB_NAME"] = model

        # 0. register all default configs
        try:
            default_configs = getattr(self, run_type + "_configs")
            CONFIGS.update(default_configs)
        except Exception:
            raise RuntimeError(f"Configs for run type {run_type} is not implemented")

        # 1. override configs if it is in debug
        if DEBUG_MODE:
            with HeadLine.wrap("RUNING IN DEBUG MODE", level=2, warning=True):
                configs = {k: v[1] for k, v in CONFIGS.items() if isinstance(v, tuple) and len(v) == 2}
                CONFIGS.verbose_update(configs)
        else:
            configs = {k: v[0] for k, v in CONFIGS.items() if isinstance(v, tuple) and len(v) == 2}
            CONFIGS.update(**configs)

        # 2. override configs if any system configs
        if len(remaining_args) > 0:
            with HeadLine.wrap("PARSING SYSTEM CONFIGS", level=3):
                CONFIGS.update_from_list(remaining_args, verbose=True)

        # 3. override configs by post process
        with HeadLine.wrap("POST PROCESS CONFIGS", level=3):
            self.post_process_configs()

    def save_dir(self, task=None, model=None):
        task = self.task if task is None else task
        model = self.model if model is None else model

        path = Path(EXPERIMENT_DIR, self.task_group, task, model).as_posix()
        path += self.kwargs.get("save_dir_suffix", "")
        if DEBUG_MODE:
            path += "-debug"
        return path

    @property
    def train_configs(self):
        # we set some global train configs here
        configs = {
            # common
            "--log-interval": ("100", "10"),
            "--log-format": "simple",
            "--log-file": "{save_dir}/log.txt",
            "--tensorboard-logdir": "{save_dir}/tensorboard",
            "--wandb-project": None if WANDB_DISABLED else "fairseq",
            "--seed": "19491001",
            "--fp16": True,
            "--memory-efficient-fp16": False,  # will increase training time
            "--on-cpu-convert-precision": True,
            # distributed
            "--ddp-backend": "c10d",
            "--gradient-as-bucket-view": True,
            # dataset
            "--num-workers": "0",
            "--required-batch-size-multiple": "1",  # keep this, otherwise the last batch will be dropped
            "--grouped-shuffling": True,
            "--fixed-validation-seed": "7",
            # checkpoint
            "--save-dir": "{save_dir}",
        }
        return configs

    def post_process_configs(self):
        _grouped_configs = {k: CONFIGS.pop(k) for k in list(CONFIGS.keys()) if "." in k}

        grouped_configs = defaultdict(dict)
        for k, v in _grouped_configs.items():
            k1, k2 = k.split(".")
            if "@" in k2:
                k2, dtype = k2.split("@")
                v = eval(dtype)(v)
            grouped_configs[k1][k2] = v

        # during training, optimizer and lr-scheduler are always required
        optimizer = CONFIGS.get("--optimizer")
        if optimizer:
            logger.warning(HeadLine.header(f"optimizer = {optimizer}", level=2))
            CONFIGS.verbose_update(grouped_configs.pop(optimizer))

        lr_scheduler = CONFIGS.get("--lr-scheduler")
        if lr_scheduler:
            logger.warning(HeadLine.header(f"lr_scheduler = {lr_scheduler}", level=2))
            CONFIGS.verbose_update(grouped_configs.pop(lr_scheduler))

        model_overrides = {}
        if grouped_configs.get("--model-overrides", None) is not None:
            model_overrides["model"] = grouped_configs.pop("--model-overrides")
        if grouped_configs.get("--task-overrides", None) is not None:
            model_overrides["task"] = grouped_configs.pop("--task-overrides")
        if len(model_overrides) > 0:
            CONFIGS.verbose_update({"--model-overrides": json.dumps(model_overrides)})

        self._post_process_configs(grouped_configs)

        if len(grouped_configs) > 0:
            logger.warning(
                f"Found unused argument groups: {dict(grouped_configs)}. "
                f"Please check if these arguments are necessary or mistakenly provided.",
            )

    def _post_process_configs(self, grouped_configs):
        save_dir = Path(self.save_dir()).as_posix()

        logger.info("Output directory: " + save_dir)
        os.makedirs(save_dir, exist_ok=True)

        for key, value in CONFIGS.items():
            if isinstance(value, str) and "{save_dir}" in value:
                CONFIGS.update({key: value.format(save_dir=save_dir)})


@Registry.register_run_type("train")
def fairseq_train(input_args) -> None:
    from fairseq_cli.train import cli_main

    cli_main(input_args=input_args)


@Registry.register_run_type("generate")
def fairseq_generate(input_args) -> None:
    from fairseq_cli.generate import cli_main

    cli_main(input_args=input_args)

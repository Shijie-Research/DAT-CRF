import argparse
import importlib
import logging
import os
import sys
import time
from pathlib import Path

import torch


def prepare_logging(level=logging.INFO) -> None:
    root_logger = logging.getLogger()

    from colorama import Fore, Style

    class ColorFormatter(logging.Formatter):
        colors = {
            "DEBUG": Fore.LIGHTBLACK_EX,
            "INFO": Style.RESET_ALL,  # key the info style
            "WARNING": Fore.YELLOW,
            "ERROR": Fore.RED,
            "CRITICAL": Fore.RED + Style.BRIGHT,
        }

        def format(self, record):
            log_fmt = self.colors.get(record.levelname) + self._fmt
            return logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S").format(record)

    class ErrorFilter(logging.Filter):
        """
        Filters out everything that is at the ERROR level or higher. This is meant to be used
        with a stdout handler when a stderr handler is also configured. That way ERROR
        messages aren't duplicated.
        """

        def filter(self, record):
            return record.levelno < logging.ERROR

    # create handlers
    formatter = ColorFormatter("%(levelname)s | %(asctime)s | %(name)s:%(lineno)d | %(message)s")
    stdout_handler: logging.Handler = logging.StreamHandler(sys.stdout)
    stderr_handler: logging.Handler = logging.StreamHandler(sys.stderr)

    handler: logging.Handler
    for handler in [stdout_handler, stderr_handler]:
        handler.setFormatter(formatter)

    # Remove the already set handlers in root logger.
    # Not doing this will result in duplicate log messages
    root_logger.handlers.clear()

    stdout_handler.setLevel(level)
    stdout_handler.addFilter(ErrorFilter())  # Make sure errors only go to stderr
    stderr_handler.setLevel(logging.ERROR)
    root_logger.setLevel(level)

    # put all the handlers on the root logger
    root_logger.addHandler(stdout_handler)
    root_logger.addHandler(stderr_handler)


def import_module_and_submodules(package_name: str) -> None:
    """Import all submodules under the given package."""
    module = importlib.import_module(package_name)

    path = getattr(module, "__path__", [])

    if len(path) > 0:
        for file in os.listdir(path[0]):
            if not file.startswith(("_", ".")):
                name = file[: file.find(".py")] if file.endswith(".py") else file
                filepath = os.path.join(path[0], name)
                if os.path.isdir(filepath) or file.endswith(".py"):
                    subpackage = package_name + "." + name
                    import_module_and_submodules(subpackage)


def main():
    base_path = Path(__file__).resolve().as_posix()

    # ensure `base_path` is the first in `sys.path`
    try:
        sys.path.pop(sys.path.index(base_path))
    except:
        pass

    sys.path.insert(0, base_path)

    tik = time.time()
    importlib.invalidate_caches()
    import_module_and_submodules("awesome_fairseq")

    from awesome_fairseq import DEBUG_MODE, HeadLine

    prepare_logging(level=logging.DEBUG if DEBUG_MODE else logging.INFO)

    logger = logging.getLogger("run")

    logger.info(f"import all scripts in {time.time() - tik:.2f} (s)")

    with HeadLine.wrap("CUDA ENVIRONMENT", level=2):
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory
            logger.info(f"Device {i}: {torch.cuda.get_device_name(i)} ({total_memory / (1024 ** 3):.2f} GB)")

    # register command and add model to command
    parser = argparse.ArgumentParser(allow_abbrev=False)
    subparsers = parser.add_subparsers()

    from awesome_fairseq import CONFIGS, MetaClass, Registry

    for run_type, models_and_tasks in Registry._model_task_registry.items():
        run_type_parser = subparsers.add_parser(run_type, allow_abbrev=False)
        model_subparsers = run_type_parser.add_subparsers()

        for model, tasks in models_and_tasks.items():
            model_parser = model_subparsers.add_parser(model, allow_abbrev=False)
            task_subparsers = model_parser.add_subparsers()

            for task, task_cls in tasks.items():
                task_parser = task_subparsers.add_parser(task, allow_abbrev=False, add_help=False)
                task_parser.set_defaults(task_cls=task_cls)

    parsed_args, remaining_args = parser.parse_known_args(sys.argv[1:])

    MetaClass.task_parser = argparse.ArgumentParser(
        prog=f"run.py {sys.argv[1]} {sys.argv[2]} {sys.argv[3]} ",
        allow_abbrev=False,
        add_help=False,
        argument_default=argparse.SUPPRESS,
    )

    # initialize task_cls, which will register all configs
    parsed_args.task_cls(run_type=sys.argv[1], model=sys.argv[2], task=sys.argv[3], remaining_args=remaining_args)

    run_main = Registry.get_run_main(sys.argv[1])
    run_main(CONFIGS.convert_to_list())


if __name__ == "__main__":
    main()

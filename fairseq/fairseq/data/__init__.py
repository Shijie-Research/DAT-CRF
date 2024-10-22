# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import importlib
import inspect
import os

from .dictionary import Dictionary, TruncatedDictionary  # noqa
from .fairseq_dataset import FairseqDataset, FairseqIterableDataset  # noqa
from .base_wrapper_dataset import BaseWrapperDataset  # noqa

datasets_dir = os.path.dirname(__file__)

for file in sorted(os.listdir(datasets_dir)):
    path = os.path.join(datasets_dir, file)
    if not file.startswith(("_", ".")) and file.endswith(".py"):
        dataset_name = file[: file.find(".py")]
        module = importlib.import_module("fairseq.data." + dataset_name)

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if name.endswith(("Dataset", "Iterator")) and obj.__module__ == f"fairseq.data.{dataset_name}":
                globals()[name] = obj

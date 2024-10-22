# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import importlib
import inspect
import os

from .layer_norm import LayerNorm  # noqa
from .positional_embedding import PositionalEmbedding  # noqa

modules_dir = os.path.dirname(__file__)

for file in sorted(os.listdir(modules_dir)):
    path = os.path.join(modules_dir, file)
    if not file.startswith(("_", ".")) and file.endswith(".py"):
        module_name = file[: file.find(".py")]
        module = importlib.import_module("fairseq.modules." + module_name)

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == f"fairseq.modules.{module_name}":
                globals()[name] = obj

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Optional

from fairseq import file_utils
from fairseq.data.encoders import register_bpe
from fairseq.dataclass import FairseqDataclass


@dataclass
class SubwordNMTBPEConfig(FairseqDataclass):
    bpe_codes: Optional[str] = field(default=None, metadata={"help": "path to subword NMT BPE"})
    bpe_separator: str = field(default="@@", metadata={"help": "BPE separator"})


@register_bpe("subword_nmt", dataclass=SubwordNMTBPEConfig)
class SubwordNMTBPE(object):
    def __init__(self, cfg):
        self.bpe_codes = cfg.bpe_codes
        self.separator = cfg.bpe_separator

    def _setup_bpe(self):
        if self.bpe_codes is None:
            raise ValueError("--bpe-codes is required for --bpe=subword_nmt")
        codes = file_utils.cached_path(self.bpe_codes)
        try:
            from subword_nmt import apply_bpe

            bpe_parser = apply_bpe.create_parser()
            bpe_args = bpe_parser.parse_args(
                [
                    "--codes",
                    codes,
                    "--separator",
                    self.bpe_separator,
                ],
            )
            self.bpe = apply_bpe.BPE(
                bpe_args.codes,
                bpe_args.merges,
                bpe_args.separator,
                None,
                bpe_args.glossaries,
            )
        except ImportError:
            raise ImportError("Please install subword_nmt with: pip install subword-nmt")

    def encode(self, x: str) -> str:
        if self.bpe is None:
            self._setup_bpe()
        return self.bpe.process_line(x)

    def decode(self, x: str) -> str:
        return (x + " ").replace(self.separator + " ", "").rstrip()

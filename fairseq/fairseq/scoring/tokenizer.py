# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unicodedata
from typing import Optional

import sacrebleu as sb

from fairseq.dataclass import ChoiceEnum


class EvaluationTokenizer(object):
    """A generic evaluation-time tokenizer, which leverages built-in tokenizers
    in sacreBLEU (https://github.com/mjpost/sacrebleu). It additionally provides
    lowercasing, punctuation removal and character tokenization, which are
    applied after sacreBLEU tokenization.

    Args:
        tokenizer_type (str): the type of sacreBLEU tokenizer to apply.
        lowercase (bool): lowercase the text.
        punctuation_removal (bool): remove punctuation (based on unicode
        category) from text.
        character_tokenization (bool): tokenize the text to characters.
    """

    SPACE = chr(32)
    SPACE_ESCAPE = chr(9601)
    _ALL_TOKENIZER_TYPES = sb.BLEU.TOKENIZERS
    ALL_TOKENIZER_TYPES = ChoiceEnum(_ALL_TOKENIZER_TYPES)

    def __init__(
        self,
        trg_lang: Optional[str] = None,
        tokenizer_type: Optional[str] = None,
        lowercase: bool = False,
        punctuation_removal: bool = False,
        character_tokenization: bool = False,
    ):
        assert (
            tokenizer_type is None or tokenizer_type in self._ALL_TOKENIZER_TYPES
        ), f"{tokenizer_type}, {self._ALL_TOKENIZER_TYPES}"

        if tokenizer_type is None:
            assert trg_lang is not None, "trg_lang must be specified if tokenizer_type is not given"

        self.lowercase = lowercase
        self.punctuation_removal = punctuation_removal
        self.character_tokenization = character_tokenization
        self.tokenizer = sb.BLEU(trg_lang=trg_lang, tokenize=tokenizer_type).tokenizer

    @classmethod
    def remove_punctuation(cls, sent: str):
        """Remove punctuation based on Unicode category."""
        return cls.SPACE.join(t for t in sent.split(cls.SPACE) if not all(unicodedata.category(c)[0] == "P" for c in t))

    def tokenize(self, sent: str):
        tokenized = self.tokenizer(sent)

        if self.punctuation_removal:
            tokenized = self.remove_punctuation(tokenized)

        if self.character_tokenization:
            tokenized = self.SPACE.join(list(tokenized.replace(self.SPACE, self.SPACE_ESCAPE)))

        if self.lowercase:
            tokenized = tokenized.lower()

        return tokenized

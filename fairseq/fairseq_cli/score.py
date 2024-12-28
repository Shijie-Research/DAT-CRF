#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
BLEU scoring of generated translations against reference translations.
"""

import argparse
import os
import sys

from fairseq.data import dictionary
from fairseq.scoring import bleu


def get_parser():
    parser = argparse.ArgumentParser(description="Command-line script for BLEU scoring.")
    # fmt: off
    parser.add_argument('-s', '--sys', default='-', help='system output')
    parser.add_argument('-r', '--ref', required=True, help='references')
    parser.add_argument(
        '-o', '--order', default=4, metavar='N',
        type=int, help='consider ngrams up to this order',
    )
    parser.add_argument(
        '--ignore-case', action='store_true',
        help='case-insensitive scoring',
    )
    parser.add_argument("--src-lang", default=None, metavar="SRC", help="source language")
    parser.add_argument("--tgt-lang", default=None, metavar="TGT", help="target language")
    parser.add_argument(
        '--sacrebleu', action='store_true',
        help='score with sacrebleu',
    )
    parser.add_argument("--sacrebleu-tokenizer", default=None, help="sacrebleu tokenizer (default: None)")
    parser.add_argument(
        '--sentence-bleu', action='store_true',
        help='report sentence-level BLEUs (i.e., with +1 smoothing)',
    )
    parser.add_argument("--bert-score", action="store_true", help="score with bertscore")
    parser.add_argument(
        "--bert-score-type", default="F",
        choices=["P", "R", "F"], help="bertscore type (default: F)",
    )
    parser.add_argument("--bert-score-rescale", action="store_true", help="rescale bertscore")
    # fmt: on
    return parser


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    assert args.sys == "-" or os.path.exists(args.sys), "System output file {} does not exist".format(args.sys)
    assert os.path.exists(args.ref), "Reference file {} does not exist".format(args.ref)

    dict = dictionary.Dictionary()

    def readlines(fd):
        for line in fd.readlines():
            if args.ignore_case:
                yield line.lower()
            else:
                yield line

    if args.sacrebleu:

        def score(fdsys):
            with open(args.ref) as fdref:
                scorer = bleu.SacrebleuScorer(
                    bleu.SacrebleuConfig(
                        target_lang=args.tgt_lang,
                        sacrebleu_tokenizer=args.sacrebleu_tokenizer,
                    ),
                )
                for i, (sys_str, ref_str) in enumerate(zip(readlines(fdsys), readlines(fdref))):
                    scorer.add_string(ref_str, sys_str)
                print(scorer.result_string(args.order))

    elif args.sentence_bleu:

        def score(fdsys):
            with open(args.ref) as fdref:
                scorer = bleu.Scorer(bleu.BleuConfig(pad=dict.pad(), eos=dict.eos(), unk=dict.unk()))
                for i, (sys_tok, ref_tok) in enumerate(zip(readlines(fdsys), readlines(fdref))):
                    scorer.reset(one_init=True)
                    sys_tok = dict.encode_line(sys_tok)
                    ref_tok = dict.encode_line(ref_tok)
                    scorer.add(ref_tok, sys_tok)
                    print(i, scorer.result_string(args.order))

    elif args.bert_score:

        from fairseq.scoring import bertscore

        def score(fdsys):
            with open(args.ref) as fdref:
                scorer = bertscore.BertScoreScorer(
                    bertscore.BertScoreScorerConfig(
                        bert_score_type=args.bert_score_type,
                        bert_score_lang=args.tgt_lang,
                        bert_score_rescale=args.bert_score_rescale,
                    ),
                )
                for sys_str, ref_str in zip(readlines(fdsys), readlines(fdref)):
                    scorer.add_string(ref_str, sys_str)
                print(scorer.result_string())

    else:

        def score(fdsys):
            with open(args.ref) as fdref:
                scorer = bleu.Scorer(
                    bleu.BleuConfig(
                        pad=dict.pad(),
                        eos=dict.eos(),
                        unk=dict.unk(),
                    ),
                )
                for sys_tok, ref_tok in zip(readlines(fdsys), readlines(fdref)):
                    sys_tok = dict.encode_line(sys_tok)
                    ref_tok = dict.encode_line(ref_tok)
                    scorer.add(ref_tok, sys_tok)
                print(scorer.result_string(args.order))

    if args.sys == "-":
        score(sys.stdin)
    else:
        with open(args.sys, "r") as f:
            score(f)


if __name__ == "__main__":
    cli_main()

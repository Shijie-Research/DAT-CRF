import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat.nonautoregressive_transformer import (
    NATransformerDecoder,
    NATransformerModel,
    base_architecture,
    init_bert_params,
)
from fairseq.models.transformer import Linear
from fairseq.modules.positional_embedding import PositionalEmbedding

from ..criterions.utilities import get_anneal_value, parse_anneal_argument

logger = logging.getLogger(__name__)

decode_strategy = [
    "greedy",
    "lookahead",
    "single-viterbi",
    "full-viterbi",
    "beamsearch",
    "full-crf",
]


def logsumexp(x: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    # Solving nan issue when x contains -inf
    # See https://github.com/pytorch/pytorch/issues/31829
    m, _ = x.max(dim=dim, keepdim=True)
    mask = m == float("-inf")
    m = m.detach()
    s = (x - m.masked_fill_(mask, 0)).exp_().sum(dim=dim, keepdim=True)
    s = s.masked_fill_(mask, 1).log_() + m.masked_fill_(mask, float("-inf"))
    return s if keepdim else s.squeeze(dim)


def get_best_alignment(emission_lprobs, transit_lprobs, prev_output_mask, target_mask):
    bsz, pre_len, tgt_len = emission_lprobs.size()

    with torch.enable_grad():
        emission_lprobs.requires_grad_()

        cumulative_lprobs = emission_lprobs.new_full((bsz, pre_len), float("-inf"))
        # always emitting BOS at the first position
        cumulative_lprobs[:, 0] = emission_lprobs[:, 0, 0]

        for t in range(1, tgt_len):
            lprobs_t = cumulative_lprobs[:, :, None] + transit_lprobs  # [bsz, pre_len, 1] + [bsz, pre_len, pre_len]
            lprobs_t = lprobs_t.max(dim=1)[0]
            lprobs_t += emission_lprobs[:, :, t]  # [bsz, pre_len]

            # if to the current position is invalid, we keep previous state
            cumulative_lprobs = torch.where(target_mask[:, [t]], lprobs_t, cumulative_lprobs)

        eos_idx = prev_output_mask.sum(-1, keepdim=True) - 1
        cumulative_lprobs = cumulative_lprobs.gather(1, eos_idx).squeeze(1)

        match_grad = torch.autograd.grad(cumulative_lprobs.sum(), [emission_lprobs])[0]

    best_alignment = match_grad.max(dim=1)[1]

    return best_alignment


def init_beam_search(*args):
    import dag_search  # noqa

    dag_search.beam_search_init(*args)


def call_dag_search(*args):
    import dag_search  # noqa

    res, score = dag_search.dag_search(*args)
    output_tokens = torch.tensor(res)
    output_scores = torch.tensor(score).unsqueeze(-1).expand_as(output_tokens)
    return output_tokens, output_scores


def subprocess_init(n):
    time.sleep(10)  # Do something to wait all subprocess to start
    print(f"overlapped decoding: subprocess {n} initializing", flush=True)
    return n


@register_model("dacrf_transformer")
class DACRFTransformerModel(NATransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        self.glat_scheduler = parse_anneal_argument(args.glance_p)

        if self.args.decode_strategy == "beamsearch":
            self.init_beam_search()

        # used for CRF finetuning
        if getattr(args, "crf_finetuning", False):
            for param in self.parameters():
                param.requires_grad = False

            self.crf_embed_query = nn.Embedding(len(self.tgt_dict), args.crf_lowrank_approx, padding_idx=self.pad)
            self.crf_embed_key = nn.Embedding(len(self.tgt_dict), args.crf_lowrank_approx, padding_idx=self.pad)

        self.register_buffer(
            "length_penalty",
            torch.arange(self.decoder.max_positions()) ** args.decode_alpha,
            persistent=False,
        )

    def init_beam_search(self):
        if self.args.decode_max_workers >= 1:  # overlapped decoding
            import concurrent
            import multiprocessing as mp

            ctx = mp.get_context("spawn")
            self.executor = concurrent.futures.ProcessPoolExecutor(  # noqa
                max_workers=self.args.decode_max_workers,
                mp_context=ctx,
                initializer=init_beam_search,
                initargs=(
                    self.args.decode_max_batchsize,
                    self.args.decode_beamsize,
                    self.args.decode_top_cand_n,
                    self.decoder.max_positions(),
                    self.args.max_decoder_batch_tokens,
                    self.args.decode_threads_per_worker,
                    self.tgt_dict,
                    self.args.decode_lm_path,
                ),
            )
            for x in self.executor.map(subprocess_init, range(self.args.decode_max_workers)):
                pass
        else:  # vanilla decoding
            init_beam_search(
                self.args.decode_max_batchsize,
                self.args.decode_beamsize,
                self.args.decode_top_cand_n,
                self.decoder.max_positions(),
                self.args.max_decoder_batch_tokens,
                self.args.decode_threads_per_worker,
                self.tgt_dict,
                self.args.decode_lm_path,
            )

    @staticmethod
    def add_args(parser):
        NATransformerModel.add_args(parser)

        parser.add_argument(
            "--upsample-scale",
            type=int,
            default=2,
            help="Specifies the upsample scale for the decoder input length in training.",
        )
        parser.add_argument(
            "--upsample-target",
            action="store_true",
            help="Set this to true will upsample target sentence length during training and predicted length during inference. "
            "This requires `--length-pred-factor > 0`",
        )
        parser.add_argument(
            "--decode-strategy",
            type=str,
            default="lookahead",
            choices=decode_strategy,
            help="Decoding strategy to use.",
        )
        parser.add_argument(
            "--glance-p",
            type=str,
            default=None,
            help="Set the glancing probability and its annealing schedule.",
        )
        # used for CRF finetuning
        parser.add_argument("--crf-finetuning", action="store_true", help="Only finetuning the CRF layer.")
        parser.add_argument("--crf-dropout", type=float, default=0.1, help="Dropout rate for CRF layer.")
        parser.add_argument("--crf-disabled", action="store_true", help="Disable CRF decoding.")
        parser.add_argument("--crf-lowrank-approx", type=int, default=64, help="Low-rank approximation for CRF layer.")
        parser.add_argument("--crf-beam-approx", type=int, default=64, help="Beam size for the CRF normalizing factor")
        parser.add_argument("--crf-decode-beam", type=int, default=8, help="Beam size for CRF decoding")

        # decode arguments
        parser.add_argument(
            "--decode-alpha",
            type=float,
            default=1.1,
            help="Parameter used for length penalty in beamsearch decoding. "
            "The sentence with the highest score is found using: 1 / |Y|^{alpha} [ log P(Y) + gamma log P_{n-gram}(Y)]",
        )
        parser.add_argument("--decode-beta", type=float, default=1, help="Parameter used to scale the token logits.")
        parser.add_argument("--decode-dedup", action="store_true", help="Enable token deduplication.")

        # other arguments
        parser.add_argument(
            "--decode-max-workers",
            type=int,
            default=0,
            help="Number of multiprocess workers to use during beamsearch decoding. "
            "More workers will consume more memory. It does not affect decoding latency but decoding throughtput, "
            'so you must use "fariseq-fastgenerate" to enable the overlapped decoding to tell the difference.',
        )
        parser.add_argument(
            "--decode-max-batchsize",
            type=int,
            default=32,
            help="Maximum batch size to use during beamsearch decoding. "
            "Should not be smaller than the actual batch size, as it is used for memory allocation.",
        )
        parser.add_argument("--decode-beamsize", type=float, default=100, help="Beam size used in beamsearch decoding.")
        parser.add_argument(
            "--decode-top-cand-n",
            type=float,
            default=5,
            help="Number of top candidates to consider during transition. "
            "This argument is used in lookahead decoding with n-gram prevention, and sample and beamsearch decoding methods.",
        )
        parser.add_argument(
            "--decode-max-beam-per-length",
            type=float,
            default=10,
            help="Maximum number of beams with the same length in each step during beamsearch decoding.",
        )
        parser.add_argument(
            "--max-encoder-batch-tokens",
            type=int,
            default=None,
            help="Specifies the maximum number of tokens for the encoder input to avoid running out of memory. "
            "The default value of None indicates no limit.",
        )
        parser.add_argument(
            "--max-decoder-batch-tokens",
            type=int,
            default=None,
            help="Specifies the maximum number of tokens for the decoder input to avoid running out of memory. "
            "The default value of None indicates no limit.",
        )
        parser.add_argument(
            "--decode-lm-path",
            type=str,
            default=None,
            help="Path to n-gram language model to use during beamsearch decoding. Set to None to disable n-gram LM.",
        )

        parser.add_argument(
            "--decode-gamma",
            type=float,
            default=0.1,
            help="Parameter used for n-gram language model score in beamsearch decoding. The sentence with the highest score "
            "is found using: 1 / |Y|^{alpha} [ log P(Y) + gamma log P_{n-gram}(Y)]",
        )

        parser.add_argument(
            "--decode-top-p",
            type=float,
            default=0.9,
            help="Maximum probability of top candidates to consider during transition. "
            "This argument is used in lookahead decoding with n-gram prevention, and sample and beamsearch decoding methods.",
        )
        parser.add_argument(
            "--decode-threads-per-worker",
            type=int,
            default=4,
            help="Number of threads per worker to use during beamsearch decoding. "
            "This setting also applies to both vanilla decoding and overlapped decoding. "
            "A value between 2 and 8 is typically optimal.",
        )

        parser.add_argument(
            "--decode-no-consecutive-repeated-ngram",
            type=int,
            default=0,
            help="Prevent consecutive repeated k-grams (k <= n) in the generated text. "
            "Use 0 to disable this feature. This argument is used in greedy, lookahead, sample, and beam search decoding methods.",
        )
        parser.add_argument(
            "--decode-no-repeated-ngram",
            type=int,
            default=0,
            help="Prevent repeated k-grams (not necessarily consecutive) with order n or higher in the generated text. "
            "Use 0 to disable this feature. This argument is used in lookahead, sample, and beam search decoding methods.",
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = DACRFTransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    @property
    def allow_ensemble(self):
        return False

    def extract_features(self, encoder_out, prev_output_tokens, **kwargs):
        features, _ = self.decoder.extract_features(prev_output_tokens, encoder_out=encoder_out, embedding_copy=False)
        word_ins_out = self.output_layer(features)
        return word_ins_out, features

    @torch.no_grad()
    def get_best_alignment(self, encoder_out, prev_output_tokens, target_tokens):
        bsz, pre_len = prev_output_tokens.size()

        target_mask = target_tokens.ne(self.pad)
        prev_output_mask = prev_output_tokens.ne(self.pad)

        emission_scores, features = self.extract_features(encoder_out, prev_output_tokens)
        emission_lprobs = F.log_softmax(emission_scores, dim=-1)

        transit_lprobs = self.decoder.compute_transit_lprobs(features, prev_output_tokens)

        best_alignment = get_best_alignment(
            emission_lprobs.gather(2, target_tokens[:, None, :].expand(-1, pre_len, -1)),
            transit_lprobs,
            prev_output_mask,
            target_mask,
        )
        return emission_lprobs, best_alignment

    @torch.no_grad()
    def glancing_sampling(self, encoder_out, prev_output_tokens, target_tokens, glat_p):
        target_mask = target_tokens.ne(self.pad)
        prev_output_mask = prev_output_tokens.ne(self.pad)
        nonspecial_mask = prev_output_mask & prev_output_tokens.ne(self.bos) & prev_output_tokens.ne(self.eos)

        emission_lprobs, best_alignment = self.get_best_alignment(encoder_out, prev_output_tokens, target_tokens)

        # if target tokens have pad, it will match to the position of BOS, so fix it.
        oracle_predictions = emission_lprobs.max(dim=-1)[1]
        oracle_predictions = torch.where(nonspecial_mask, oracle_predictions, prev_output_tokens)

        scattered_predictions = oracle_predictions.scatter(1, best_alignment, target_tokens)
        scattered_predictions = torch.where(nonspecial_mask, scattered_predictions, prev_output_tokens)

        scattered_mask = torch.zeros_like(prev_output_tokens).scatter(1, best_alignment, 1).bool()
        scattered_mask &= nonspecial_mask

        unmatched_num = (oracle_predictions != scattered_predictions).sum(1, keepdim=True)  # noqa

        probs = torch.zeros_like(oracle_predictions).float().uniform_().masked_fill_(~scattered_mask, -2)
        probs_thresh = probs.sort(descending=True)[0].gather(-1, (unmatched_num * glat_p + 0.5).long())
        # only positions whose probs are higher than the threshold will be replaced by the prediction
        keep_mask = probs <= probs_thresh

        glat_prev_output_tokens = scattered_predictions.clone()
        glat_prev_output_tokens[keep_mask] = prev_output_tokens[keep_mask]

        total = utils.item((target_mask.sum(-1) - 2).sum())
        n_correct = total - utils.item(unmatched_num.sum())
        glat_info = {"glat_total": total, "glat_correct": n_correct, "glat_p": glat_p}

        return glat_prev_output_tokens, glat_info

    def forward(self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs):
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # in this case, input length was determinate by the upsample scale
        # to avoid adding too many padding tokens, we use the same scale for all sentences
        if getattr(self.args, "upsample_target", False):
            length_tgt = tgt_tokens.ne(self.pad).sum(1) * int(self.args.upsample_scale)
        else:
            length_tgt = src_lengths * int(self.args.upsample_scale)

        prev_output_tokens = self.initialize_output_tokens(encoder_out, src_tokens, length_tgt).output_tokens

        glat_p = get_anneal_value(self.glat_scheduler, self.get_num_updates())
        if glat_p > 0:
            prev_output_tokens, self.glat_info = self.glancing_sampling(
                encoder_out,
                prev_output_tokens=prev_output_tokens,
                target_tokens=tgt_tokens,
                glat_p=glat_p,
            )
        else:
            self.glat_info = {}

        # decoding
        emission_scores, features = self.extract_features(encoder_out, prev_output_tokens)
        emission_lprobs = F.log_softmax(emission_scores, dim=-1)

        transit_lprobs = self.decoder.compute_transit_lprobs(features, prev_output_tokens)

        if not getattr(self.args, "crf_finetuning", False):  # compute dag loss
            dag_loss = self._compute_dag_loss(
                prev_output_tokens,
                tgt_tokens,
                emission_lprobs=emission_lprobs,
                transit_lprobs=transit_lprobs,
            )
            ret = {"dag_loss": {"loss": dag_loss}}
        else:
            dacrf_loss = self._compute_dacrf_loss(
                prev_output_tokens,
                tgt_tokens,
                emission_lprobs=emission_lprobs,
                transit_lprobs=transit_lprobs,
            )
            ret = {"dacrf_loss": {"loss": dacrf_loss}}

        # length prediction
        if self.args.length_loss_factor > 0:
            # length prediction
            length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
            length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)
            ret["length"] = {"out": length_out, "tgt": length_tgt, "factor": self.args.length_loss_factor}

        return ret

    def _compute_dag_loss(self, prev_output_tokens, target_tokens, emission_lprobs, transit_lprobs):
        bsz, pre_len = prev_output_tokens.size()
        tgt_len = target_tokens.size(-1)

        # only consider the target tokens
        emission_lprobs = emission_lprobs.gather(2, target_tokens[:, None, :].expand(-1, pre_len, -1))

        target_mask = target_tokens.ne(self.pad)
        prev_output_mask = prev_output_tokens.ne(self.pad)

        # set the cumulative lprobs
        cumulative_lprobs = emission_lprobs.new_full((bsz, pre_len), float("-inf"))
        cumulative_lprobs[:, 0] = emission_lprobs[:, 0, 0]

        for t in range(1, tgt_len):
            # [bsz, pre_len, 1] + [bsz, pre_len, pre_len]
            lprobs_t = cumulative_lprobs[:, :, None] + transit_lprobs
            lprobs_t = logsumexp(lprobs_t, dim=1)
            lprobs_t += emission_lprobs[:, :, t]  # [bsz, 1, pre_len]

            # only compute the ground-truth path
            cumulative_lprobs = torch.where(target_mask[:, [t]], lprobs_t, cumulative_lprobs)

        eos_idx = prev_output_mask.sum(-1, keepdim=True) - 1
        dag_loss = -cumulative_lprobs.gather(1, eos_idx).squeeze(1)
        target_mask = target_mask.type_as(dag_loss)

        invalid_masks = dag_loss.isnan()
        if invalid_masks.sum() > 0:
            logger.warning(f"{invalid_masks.sum()} samples have nan dag_loss.")
            dag_loss = dag_loss[~invalid_masks]
            target_mask = target_mask[~invalid_masks]

        invalid_masks = dag_loss.isinf()
        if invalid_masks.sum() > 0:
            logger.warning(
                f"{invalid_masks.sum()} samples have inf loss, "
                f"which usually happens when the input length is smaller than the actual output length. "
                f"Please use a larger upsample value!",
            )
            dag_loss = dag_loss[~invalid_masks]
            target_mask = target_mask[~invalid_masks]

        return (dag_loss / target_mask.sum(-1)).mean()

    def _compute_dacrf_loss(self, prev_output_tokens, target_tokens, emission_lprobs, transit_lprobs):
        bsz, tgt_len = target_tokens.size()
        pre_len = prev_output_tokens.size(1)

        target_mask = target_tokens.ne(self.pad)
        prev_output_mask = prev_output_tokens.ne(self.pad)

        best_alignment = get_best_alignment(
            emission_lprobs.gather(2, target_tokens[:, None, :].expand(-1, pre_len, -1)),
            transit_lprobs,
            prev_output_mask,
            target_mask,
        )

        bsz_idx = torch.arange(bsz).unsqueeze(-1).expand(-1, tgt_len)
        emission_lprobs = emission_lprobs[bsz_idx, best_alignment]
        # transit_lprobs = transit_lprobs[bsz_idx[:, :-1], best_alignment[:, :-1], best_alignment[:, 1:]]

        # emission_lprobs[:, 1:] += transit_lprobs.unsqueeze(-1)

        numerator = self._compute_dacrf_numerator(emission_lprobs, target_tokens, target_mask)
        denominator = self._compute_dacrf_denominator(emission_lprobs, target_tokens, target_mask)

        dacrf_loss = -(numerator - denominator) / target_mask.type_as(numerator).sum(-1)

        return dacrf_loss.mean()

    def _compute_dacrf_numerator(self, emission_scores, targets, masks=None):
        lowrank = self.args.crf_lowrank_approx

        emission_scores = emission_scores.gather(2, targets[:, :, None]).squeeze(-1)  # B x T

        E1 = self.crf_embed_query(targets[:, :-1]).view(-1, lowrank)
        E2 = self.crf_embed_key(targets[:, 1:]).view(-1, lowrank)

        if self.training:
            E1 = F.dropout(E1, p=self.args.crf_dropout, training=True)
            E2 = F.dropout(E2, p=self.args.crf_dropout, training=True)

        crf_scores = (E1 * E2).sum(1).view(emission_scores.size(0), -1)

        emission_scores[:, 1:] += crf_scores

        emission_scores = emission_scores.masked_fill(~masks, 0)

        return emission_scores.sum(-1)

    def _compute_dacrf_denominator(self, emissions_scores, targets=None, masks=None):
        beam = self.args.crf_beam_approx
        lowrank = self.args.crf_lowrank_approx

        bsz, seq_len, _ = emissions_scores.size()

        _emissions_scores = emissions_scores.scatter(2, targets[:, :, None], float("inf"))
        beam_targets = _emissions_scores.topk(beam, 2)[1]
        beam_emission_scores = emissions_scores.gather(2, beam_targets)

        E1 = self.crf_embed_query(beam_targets[:, :-1]).view(-1, beam, lowrank)
        E2 = self.crf_embed_key(beam_targets[:, 1:]).view(-1, beam, lowrank)

        if self.training:
            E1 = F.dropout(E1, p=self.args.crf_dropout, training=True)
            E2 = F.dropout(E2, p=self.args.crf_dropout, training=True)

        beam_crf_matrix = torch.bmm(E1, E2.transpose(1, 2)).view(bsz, -1, beam, beam)

        # compute the normalizer in the log-space
        score = beam_emission_scores[:, 0]  # B x K
        for i in range(1, seq_len):
            next_score = score[:, :, None] + beam_crf_matrix[:, i - 1]
            next_score = logsumexp(next_score, dim=1) + beam_emission_scores[:, i]

            score = torch.where(masks[:, i : i + 1], next_score, score)

        # Sum (log-sum-exp) over all possible tags
        return logsumexp(score, dim=1)

    def logging_output_train_valid_hook(self, logging_outputs):
        logging_outputs.update(self.glat_info)
        self.glat_info = {}

    @classmethod
    def reduce_metrics(cls, logging_outputs):
        glat_p = logging_outputs[0].get("glat_p", None)

        if glat_p is None:
            return

        assert all(log["glat_p"] == glat_p for log in logging_outputs)

        from fairseq import metrics

        metrics.log_scalar("glat_p", glat_p, weight=0, round=3)

        glat_total = utils.item(sum(log.get("glat_total", 0) for log in logging_outputs))
        if glat_total > 0:
            glat_correct = utils.item(sum(log.get("glat_correct", 0) for log in logging_outputs))

            metrics.log_scalar("_glat_total", glat_total)
            metrics.log_scalar("_glat_correct", glat_correct)
            metrics.log_derived(
                "glat_accuracy",
                lambda meters: (
                    round(meters["_glat_correct"].sum * 100.0 / meters["_glat_total"].sum, 3)
                    if meters["_glat_total"].sum > 0
                    else float("nan")
                ),
            )

    def forward_decoder(self, decoder_out, encoder_out, src_tokens=None, decoding_format=None, **kwargs):
        upsample_target = getattr(self.args, "upsample_target", False)
        if upsample_target and int(self.args.upsample_scale) == 1:
            return super().forward_decoder(
                decoder_out,
                encoder_out,
                src_tokens=src_tokens,
                decoding_format=decoding_format,
                **kwargs,
            )

        history = decoder_out.history

        if getattr(self.args, "upsample_target", False):
            length_tgt = decoder_out.output_tokens.ne(self.pad).sum(1) * int(self.args.upsample_scale)
        else:
            length_tgt = src_tokens.ne(self.pad).sum(1) * int(self.args.upsample_scale)

        output_tokens = self.initialize_output_tokens(encoder_out, src_tokens, length_tgt).output_tokens

        emission_scores, features = self.extract_features(encoder_out, output_tokens)
        emission_lprobs = F.log_softmax(emission_scores, dim=-1)

        transit_lprobs = self.decoder.compute_transit_lprobs(features, output_tokens)

        # decoding
        if self.args.decode_strategy in ["greedy", "lookahead"]:
            inference_cls = self._inference_lookahead
        elif self.args.decode_strategy == "joint-viterbi":
            inference_cls = self._inference_joint_viterbi

        elif self.args.decode_strategy in ["single-viterbi"]:
            inference_cls = self._inference_single_viterbi

        elif self.args.decode_strategy in ["full-viterbi"]:
            inference_cls = self._inference_full_viterbi

        elif self.args.decode_strategy == "beamsearch":
            inference_cls = self._inference_beamsearch

        elif self.args.decode_strategy == "full-crf":
            inference_cls = self._inference_full_crf

        else:
            raise ValueError(f"Unknown decode strategy: {self.args.decode_strategy}")

        output_tokens, output_scores, output_aligns = inference_cls(
            emission_lprobs,
            transit_lprobs,
            output_tokens,
            decoder_out.output_tokens,
        )

        if getattr(self.args, "crf_finetuning", False) and not getattr(self.args, "crf_disabled", False):
            output_tokens = self._inference_crf(output_tokens, emission_lprobs, transit_lprobs, output_aligns)

        if history is not None:
            history.append({"tokens": output_tokens.clone(), "scores": output_scores.clone()})  # noqa

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )

    def _ensure_valid_target_mask(self, prev_output_mask, target_mask):
        pre_len = prev_output_mask.size(1)
        tgt_len = target_mask.size(1)

        invalid_batch_mask = (prev_output_mask.sum(-1) < target_mask.sum(-1)).bool()  # noqa
        if invalid_batch_mask.any():
            if tgt_len < pre_len:
                target_mask[invalid_batch_mask] = prev_output_mask[invalid_batch_mask][:, :tgt_len]
            else:
                target_mask[invalid_batch_mask] = False
                target_mask[invalid_batch_mask][:, :pre_len] = prev_output_mask[invalid_batch_mask]

    def _inference_crf(self, output_tokens, emission_lprobs, transit_lprobs, output_aligns):
        beam = self.args.crf_decode_beam
        lowrank = self.args.crf_lowrank_approx

        output_masks = output_tokens.ne(self.pad)

        emission_lprobs = emission_lprobs.gather(
            1,
            output_aligns[:, :, None].expand(-1, -1, emission_lprobs.size(-1)),
        )

        bsz, seq_len = emission_lprobs.shape[:2]
        beam_emission_lprobs, beam_targets = emission_lprobs.topk(beam, 2)

        E1 = self.crf_embed_query(beam_targets[:, :-1]).view(-1, beam, lowrank)
        E2 = self.crf_embed_key(beam_targets[:, 1:]).view(-1, beam, lowrank)
        beam_transit_matrix = torch.bmm(E1, E2.transpose(1, 2)).view(bsz, -1, beam, beam)

        # compute the normalizer in the log-space
        cumulative_lprobs = beam_emission_lprobs[:, 0]  # B x K
        dummy = utils.new_arange(cumulative_lprobs)

        traj_index = []
        for i in range(1, seq_len):
            lprobs_t = cumulative_lprobs[:, :, None] + beam_transit_matrix[:, i - 1]
            lprobs_t, index = lprobs_t.max(dim=1)
            cumulative_lprobs = lprobs_t + beam_emission_lprobs[:, i]

            if output_masks is not None:
                index = torch.where(output_masks[:, i : i + 1], index, dummy)
            traj_index.append(index)

        # now running the back-tracing and find the best
        best_index = cumulative_lprobs.max(dim=1)[1]
        finalized_tokens = [best_index[:, None]]

        for idx in reversed(traj_index):
            finalized_tokens.insert(0, idx.gather(1, finalized_tokens[0]))

        finalized_tokens = torch.cat(finalized_tokens, 1)
        finalized_tokens = beam_targets.gather(2, finalized_tokens[:, :, None])[:, :, 0]
        finalized_tokens[~output_masks] = self.pad

        return finalized_tokens

    def _inference_lookahead(self, emission_lprobs, transit_lprobs, prev_output_tokens, target_tokens):
        prev_output_mask = prev_output_tokens.ne(self.pad)

        output_lprobs, output_tokens = emission_lprobs.max(dim=-1)

        output_tokens = output_tokens.tolist()
        output_length = prev_output_mask.sum(-1).tolist()

        if self.args.decode_strategy == "lookahead":
            transit_lprobs = transit_lprobs + output_lprobs.unsqueeze(1) * self.args.decode_beta

        links_idx = transit_lprobs.max(dim=-1)[1].tolist()

        unpad_output_tokens = []
        unpad_output_aligns = []
        for i, length in enumerate(output_length):
            last = output_tokens[i][0]
            j = 0
            res = [last]
            als = [j]
            while j != length - 1:
                j = links_idx[i][j]  # noqa
                now_token = output_tokens[i][j]
                if now_token == self.tgt_dict.pad():
                    break
                if not getattr(self.args, "decode_dedup", False) or now_token != last:
                    res.append(now_token)
                    als.append(j)
                last = now_token
            unpad_output_tokens.append(res)
            unpad_output_aligns.append(als)

        output_seqlen = max([len(res) for res in unpad_output_tokens])
        output_tokens = [res + [self.pad] * (output_seqlen - len(res)) for res in unpad_output_tokens]
        output_tokens = prev_output_tokens.new_tensor(output_tokens)

        output_aligns = [als + [0] * (output_seqlen - len(als)) for als in unpad_output_aligns]
        output_aligns = prev_output_tokens.new_tensor(output_aligns)

        output_scores = torch.full_like(output_tokens, 1.0, dtype=torch.float)
        return output_tokens, output_scores, output_aligns

    def _inference_single_viterbi(self, emission_lprobs, transit_lprobs, prev_output_tokens, target_tokens):
        tgt_len = target_tokens.size(1)

        target_mask = target_tokens.ne(self.pad)
        prev_output_mask = prev_output_tokens.ne(self.pad)

        self._ensure_valid_target_mask(prev_output_mask, target_mask)

        emission_lprobs, emission_tokens = emission_lprobs.max(dim=-1)

        cumulative_lprobs = torch.full_like(emission_lprobs, float("-inf"))
        # at the first step we only consider the emission score of the first position
        cumulative_lprobs[:, 0] = emission_lprobs[:, 0]

        eos_idx = prev_output_mask.sum(-1, keepdim=True) - 1

        traj_index = []
        for t in range(1, tgt_len):  # single path only consider trajectory with len(Y)
            lprobs_t = cumulative_lprobs[:, :, None] + transit_lprobs
            lprobs_t, index_t = torch.max(lprobs_t, dim=1)
            cumulative_lprobs = lprobs_t + emission_lprobs

            # if to the current position is invalid, we set the pointer index of all tokens to eos position
            index_t = torch.where(target_mask[:, [t]], index_t, eos_idx)
            traj_index.append(index_t)

        # max_length * batch
        best_alignment = [eos_idx]
        for index in reversed(traj_index):
            best_alignment.insert(0, index.gather(1, best_alignment[0]))
        best_alignment = torch.cat(best_alignment, 1)

        output_scores = emission_lprobs.gather(1, best_alignment)
        output_tokens = emission_tokens.gather(1, best_alignment)

        return output_tokens, output_scores, best_alignment

    def _inference_full_viterbi(self, emission_lprobs, transit_lprobs, prev_output_tokens, target_tokens):
        prev_output_mask = prev_output_tokens.ne(self.pad)

        unreduced_logits, unreduced_tokens = emission_lprobs.max(dim=-1)
        unreduced_tokens = unreduced_tokens.tolist()

        batch_size, graph_length, _ = transit_lprobs.size()

        eos_idx = prev_output_mask.sum(-1, keepdim=True) - 1

        scores = []
        indexs = []
        # batch * graph_length
        alpha_t = unreduced_logits[:, 0].unsqueeze(1) + transit_lprobs[:, 0] + unreduced_logits

        # the exact max_length should be graph_length - 2, but we can reduce it to an appropriate extent to speedup decoding
        for i in range(graph_length - 2):
            alpha_t, index = torch.max(alpha_t.unsqueeze(-1) + transit_lprobs, dim=1)
            alpha_t += unreduced_logits
            scores.append(alpha_t)
            indexs.append(index)

        # max_length * batch * graph_length
        indexs = torch.stack(indexs, dim=0)
        scores = torch.stack(scores, dim=0)

        scores = scores.gather(-1, eos_idx.unsqueeze(0).expand(scores.size(0), -1, -1))[:, :, 0]

        # max_length * batch
        scores = scores / self.length_penalty[1 : graph_length - 1].unsqueeze(1)

        max_score, pred_length = torch.max(scores, dim=0)
        pred_length += 1

        indexs = indexs.tolist()
        eos_idx = eos_idx[:, 0].tolist()
        pred_length = pred_length.tolist()

        unpad_output_tokens = []
        unpad_output_aligns = []
        for i, length in enumerate(pred_length):
            j = eos_idx[i]
            last = unreduced_tokens[i][j]
            assert last == self.eos
            res = [last]
            als = [j]
            for k in reversed(range(length)):
                j = indexs[k][i][j]
                now_token = unreduced_tokens[i][j]
                if not getattr(self.args, "decode_dedup", False) or now_token != last:
                    res.insert(0, now_token)
                    als.insert(0, j)
                last = now_token
            unpad_output_tokens.append([self.bos] + res)
            unpad_output_aligns.append([0] + als)
        output_seqlen = max([len(res) for res in unpad_output_tokens])
        output_tokens = [res + [self.pad] * (output_seqlen - len(res)) for res in unpad_output_tokens]
        output_tokens = prev_output_tokens.new_tensor(output_tokens)

        output_aligns = [als + [0] * (output_seqlen - len(als)) for als in unpad_output_aligns]
        output_aligns = prev_output_tokens.new_tensor(output_aligns)

        output_scores = torch.full_like(output_tokens, 1.0, dtype=torch.float)
        return output_tokens, output_scores, output_aligns

    def _inference_beamsearch(self, emission_lprobs, transit_lprobs, prev_output_tokens, target_tokens):
        output_lengths = prev_output_tokens.ne(self.pad).sum(-1)

        batch_size, prelen, _ = transit_lprobs.shape

        assert (
            batch_size <= self.args.decode_max_batchsize
        ), "Please set --decode-max-batchsize for beamsearch with a larger batch size"

        top_logits, top_logits_idx = emission_lprobs.topk(self.args.decode_top_cand_n, dim=-1)
        dagscores_arr = transit_lprobs.unsqueeze(-1) + top_logits.unsqueeze(1) * self.args.decode_beta

        dagscores, top_cand_idx = dagscores_arr.reshape(batch_size, prelen, -1).topk(self.args.decode_top_cand_n, -1)

        nextstep_idx = torch.div(top_cand_idx, self.args.decode_top_cand_n, rounding_mode="floor")
        logits_idx_idx = top_cand_idx % self.args.decode_top_cand_n  # batch * prelen * top_cand_n
        idx1 = utils.new_arange(transit_lprobs, batch_size)[:, None, None].expand(*nextstep_idx.shape)
        logits_idx = top_logits_idx[idx1, nextstep_idx, logits_idx_idx]  # batch * prelen * top_cand_n

        if (
            dagscores.get_device() == -1
            and self.args.decode_strategy == "beamsearch"
            and self.args.decode_max_workers < 1
        ):
            raise RuntimeError(
                "Please specify decode_max_workers at least 1 if you want to run DA-Transformer on cpu while using beamsearch decoding. "
                "It will use a separate process for beamsearch because the multi-thread library used in PyTorch and DAG-Search is conflict.",
            )

        dagscores = np.ascontiguousarray(dagscores.float().cpu().numpy())
        nextstep_idx = np.ascontiguousarray(nextstep_idx.int().cpu().numpy())
        logits_idx = np.ascontiguousarray(logits_idx.int().cpu().numpy())
        output_length_cpu = np.ascontiguousarray(output_lengths.int().cpu().numpy())

        if self.args.decode_max_workers >= 1:
            future = self.executor.submit(
                call_dag_search,
                dagscores,
                nextstep_idx,
                logits_idx,
                output_length_cpu,
                self.args.decode_alpha,
                self.args.decode_gamma,
                self.args.decode_beamsize,
                self.args.decode_max_beam_per_length,
                self.args.decode_top_p,
                self.tgt_dict.pad_index,
                self.tgt_dict.bos_index,
                1 if getattr(self.args, "decode_dedup", False) else 0,
                self.args.decode_no_consecutive_repeated_ngram,
                self.args.decode_no_repeated_ngram,
            )
            return future
        else:
            output_tokens, output_scores = call_dag_search(
                dagscores,
                nextstep_idx,
                logits_idx,
                output_length_cpu,
                self.args.decode_alpha,
                self.args.decode_gamma,
                self.args.decode_beamsize,
                self.args.decode_max_beam_per_length,
                self.args.decode_top_p,
                self.tgt_dict.pad_index,
                self.tgt_dict.bos_index,
                1 if getattr(self.args, "decode_dedup", False) else 0,
                self.args.decode_no_consecutive_repeated_ngram,
                self.args.decode_no_repeated_ngram,
            )
            return output_tokens.to(prev_output_tokens.device), output_scores.to(prev_output_tokens.device), None

    def _inference_full_crf(self, emission_lprobs, transit_lprobs, prev_output_tokens, target_tokens):
        bsz, pre_len = prev_output_tokens.size()
        beam_size = self.args.crf_decode_beam

        prev_output_mask = prev_output_tokens.ne(self.pad)

        beam_lprobs, beam_tokens = emission_lprobs.topk(beam_size, dim=-1)
        dag_lprobs = beam_lprobs[:, None, :, :] + transit_lprobs[:, :, :, None]

        cumulative_lprobs = beam_lprobs.new_zeros(bsz, pre_len, beam_size).fill_(float("-inf"))
        # at the first step we only consider the emission score of the first position
        cumulative_lprobs[:, 0, 0] = beam_lprobs[:, 0, 0]

        E1 = self.crf_embed_query(beam_tokens)
        E2 = self.crf_embed_key(beam_tokens)
        crf_scores = torch.einsum("bsmd,btnd->bstmn", E1, E2)

        dag_lprobs = dag_lprobs[:, :, :, None, :] + crf_scores

        eos_idx = prev_output_mask.sum(-1) - 1

        traj_index = []
        beam_index = []
        all_lprobs = []
        for t in range(1, pre_len):
            lprobs_t = cumulative_lprobs[:, :, None, :, None] + dag_lprobs
            lprobs_t, index_t = lprobs_t.max(dim=1)
            cumulative_lprobs, index_b = lprobs_t.max(dim=2)

            # compute the length penalized scores of sentences that end here
            traj_index.append(index_t)
            beam_index.append(index_b)
            all_lprobs.append(cumulative_lprobs)

        all_lprobs = torch.stack(all_lprobs, dim=0)
        eos_lprobs = all_lprobs.gather(2, eos_idx[None, :, None, None].expand(pre_len - 1, -1, -1, beam_size)).squeeze(
            2,
        )
        eos_lprobs = eos_lprobs[:, :, 0]

        eos_lprobs = eos_lprobs / self.length_penalty[2 : pre_len + 1].unsqueeze(1)

        max_scores, length = eos_lprobs.max(dim=0)
        length += 2

        # max_length * batch
        max_length = length.max()
        traj_index = traj_index[: max_length - 1]
        beam_index = beam_index[: max_length - 1]

        best_traj_idx = eos_idx.unsqueeze(-1)  # all sentences start from eos in a reversed way
        best_traj_list = [best_traj_idx.clone()]

        best_beam_idx = eos_idx.new_zeros(bsz, 1)
        best_beam_list = [best_beam_idx.clone()]

        mask = utils.new_arange(prev_output_tokens, max_length).unsqueeze(0).expand(bsz, -1) < length.unsqueeze(-1)
        bsz_idx = torch.arange(bsz).unsqueeze(-1)

        for i, (traj, beam) in enumerate(zip(reversed(traj_index), reversed(beam_index))):
            update_mask = mask[:, [-1 - i]]
            # first get the best
            beam_idx = beam[bsz_idx, best_traj_list[0], best_beam_list[0]]
            traj_idx = traj[bsz_idx, best_traj_list[0], beam_idx, best_beam_list[0]]

            # only update sentences that are longer or equal than current length
            best_traj_idx[update_mask] = traj_idx[update_mask]
            best_traj_list.insert(0, best_traj_idx.clone())

            best_beam_idx[update_mask] = beam_idx[update_mask]
            best_beam_list.insert(0, best_beam_idx.clone())

        best_traj = torch.cat(best_traj_list, 1)
        best_beam = torch.cat(best_beam_list, 1)

        # first gather all beams in the selected path
        output_tokens = beam_tokens[bsz_idx.expand(*best_beam.shape), best_traj, best_beam]
        output_lprobs = beam_lprobs[bsz_idx.expand(*best_beam.shape), best_traj, best_beam]

        return output_tokens, output_lprobs, None

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        old_keys = list(state_dict.keys())

        # move transit-related parameters to decoder
        for key in old_keys:
            if key.startswith(("link_positional", "query_linear", "key_linear", "gate_linear")):
                new_key = "decoder." + key
                state_dict[new_key] = state_dict.pop(key)

            if key.startswith(("embed_query", "embed_key")):
                new_key = "crf_" + key
                state_dict[new_key] = state_dict.pop(key)


class DACRFTransformerDecoder(NATransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn)
        max_pos = self.max_positions()
        embed_dim = args.decoder.embed_dim

        self.link_positional = PositionalEmbedding(max_pos, embed_dim, self.padding_idx, learned=True)

        self.query_linear = Linear(embed_dim * 2, embed_dim, bias=False)
        self.key_linear = Linear(embed_dim * 2, embed_dim, bias=False)
        self.gate_linear = Linear(embed_dim * 2, args.decoder.attention_heads, bias=False)

        self.register_buffer(
            "right_triangular_mask",
            # initialize with a relatively large value
            torch.ones(128, 128).bool().triu_(1),
            persistent=False,
        )

    def valid_transition_mask(self, max_len):
        if self.right_triangular_mask.size(0) < max_len:
            self.right_triangular_mask = self.right_triangular_mask.new_ones(max_len, max_len).bool().triu_(1)
        return self.right_triangular_mask[:max_len, :max_len]

    def compute_transit_lprobs(self, features, prev_output_tokens):
        bsz, pre_len = features.shape[:2]

        num_heads = self.args.decoder.attention_heads
        head_dim = self.args.decoder.embed_dim // num_heads

        prev_output_mask = (
            prev_output_tokens.ne(self.padding_idx).unsqueeze(1).repeat(1, num_heads, 1).view(-1, pre_len)
        )
        valid_transit_mask = prev_output_mask.unsqueeze(1) & self.valid_transition_mask(pre_len)

        # [pre_len, bsz, dim]
        features = torch.cat([features, self.link_positional(prev_output_tokens)], -1).transpose(0, 1)

        # Use multiple heads in calculating transition matrix
        q_chunks = self.query_linear(features).contiguous().view(pre_len, bsz * num_heads, -1).transpose(0, 1)
        k_chunks = self.key_linear(features).contiguous().view(pre_len, bsz * num_heads, -1).transpose(0, 1)

        gate_lprobs = F.log_softmax(self.gate_linear(features).transpose(0, 1), dim=-1, dtype=torch.float)

        # Transition probability for each head, with shape batch_size * pre_len * pre_len * chunk_num
        transit_scores = torch.bmm(q_chunks.float(), k_chunks.transpose(1, 2).float()) / head_dim**0.5
        transit_scores = transit_scores.masked_fill(~valid_transit_mask, float("-inf"))

        transit_lprobs = F.log_softmax(transit_scores, dim=-1)
        transit_lprobs = transit_lprobs.masked_fill(~valid_transit_mask, float("-inf"))

        transit_lprobs = transit_lprobs.view(bsz, num_heads, pre_len, pre_len).permute(0, 2, 3, 1)

        transit_lprobs = logsumexp(transit_lprobs + gate_lprobs.unsqueeze(2), dim=-1)

        return transit_lprobs.to(dtype=features.dtype)


@register_model_architecture("dacrf_transformer", "dacrf_transformer_wmt_en_de")
def dacrf_transformer_wmt_en_de(args):
    base_architecture(args)


@register_model_architecture("dacrf_transformer", "dacrf_transformer_iwslt_de_en")
def dacrf_transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)


@register_model_architecture("dacrf_transformer", "dacrf_transformer_big")
def dacrf_transformer_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)

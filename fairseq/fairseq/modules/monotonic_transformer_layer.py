# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

from torch import Tensor

from fairseq.modules.transformer_layer import (
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)

from .monotonic_multihead_attention import build_monotonic_attention


class TransformerMonotonicEncoderLayer(TransformerEncoderLayer):
    def forward(self, x, encoder_padding_mask, **kwargs):
        seq_len, _, _ = x.size()
        attn_mask = x.new_ones([seq_len, seq_len]).triu(1)
        attn_mask = attn_mask.masked_fill(attn_mask.bool(), float("-inf"))
        return super().forward(x, encoder_padding_mask, attn_mask)


class TransformerMonotonicDecoderLayer(TransformerDecoderLayer):
    def build_encoder_attention(self, embed_dim, cfg):
        return build_monotonic_attention(cfg)

    def prune_incremental_state(self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]):
        input_buffer = self.self_attn._get_input_buffer(incremental_state)
        for key in ["prev_key", "prev_value"]:
            input_buffer_key = input_buffer[key]
            assert input_buffer_key is not None
            if input_buffer_key.size(2) > 1:
                input_buffer[key] = input_buffer_key[:, :, :-1, :]
            else:
                typed_empty_dict: Dict[str, Optional[Tensor]] = {}
                input_buffer = typed_empty_dict
                break
        assert incremental_state is not None
        self.self_attn._set_input_buffer(incremental_state, input_buffer)

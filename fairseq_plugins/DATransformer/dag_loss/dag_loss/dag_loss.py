##########################################################################
# Copyright (C) 2022 COAI @ Tsinghua University

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#         http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################


import torch
from torch.autograd import Function

try:
    import dag_loss_cuda

except ImportError:
    raise ImportError("Please install dag_loss_cuda!")


class DagLossFunc(Function):
    config = 1  # forward
    config1 = 2  # backward
    config2 = 2  # backward

    @staticmethod
    def forward(
        ctx,
        match_all,  # bsz * tarlen * prelen
        links,  # bsz * prelen * translen
        output_length,  # bsz
        target_length,  # bsz
    ):
        r"""
        Function to calculate the dag loss.
        Input:
            match_all (torch.FloatTensor or torch.HalfTensor):
                Shape: [batch_size, max_target_length, max_output_length]
                match_all[b, i, j] represents -log P(y_i| v_j), the probability of predicting the i-th token in the reference
                based on the j-th vertex.
                (Note: float32 are preferred; float16 may cause precision problem)
            links (torch.FloatTensor or torch.HalfTensor):
                Shape: [batch_size, max_output_length, max_transition_length]
                links[b, i, j] represents the transition probability from the i-th vertex to **the (i+j)-th vertex**.
                (Note: this parameter is different from the torch version)
            output_length (torch.LongTensor):
                Shape: [batch_size]
                output_length should be the graph size, the vertices (index >= graph size) are ignored
            target_length (torch.LongTensor):
                Shape: [batch_size]
                target_length is the reference length, the tokens (index >= target length) are ignored

        Output (torch.FloatTensor or torch.HalfTensor):
            Shape: [batch_size]
            the loss of each sample
        """
        require_gradient = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
        match_all = match_all.contiguous()
        links = links.contiguous()
        alpha, beta = dag_loss_cuda.dag_loss(
            match_all,
            links,
            output_length,
            target_length,
            require_gradient,
            DagLossFunc.config,
        )  # bsz * prelen * tarlen

        if require_gradient:
            res = beta[:, 0, 0].clone()
        else:
            res = alpha[range(alpha.shape[0]), target_length - 1, output_length - 1]
        ctx.save_for_backward(alpha, beta, match_all, links, output_length, target_length)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        alpha, beta, match_all, links, output_length, target_length = ctx.saved_tensors
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            grad_match_all, grad_links = dag_loss_cuda.dag_loss_backward(
                grad_output,
                alpha,
                beta,
                match_all,
                links,
                output_length,
                target_length,
                DagLossFunc.config1,
                DagLossFunc.config2,
            )
            return grad_match_all, grad_links, None, None
        else:
            return None, None, None, None


dag_loss = DagLossFunc.apply


class DagBestAlignmentFunc(Function):
    config = 1

    @staticmethod
    def forward(
        ctx,
        match_all,  # bsz * tarlen * prelen
        links,  # bsz * prelen * translen
        output_length,  # bsz
        target_length,  # bsz
    ):
        r"""
        Function to obtain the alignment between prediction and reference
        Input:
            match_all (torch.FloatTensor or torch.HalfTensor):
                Shape: [batch_size, max_target_length, max_output_length]
                match_all[b, i, j] represents -log P(y_i| v_j), the probability of predicting the i-th token in the reference
                based on the j-th vertex.
                (Note: float32 are preferred; float16 may cause precision problem)
            links (torch.FloatTensor or torch.HalfTensor):
                Shape: [batch_size, max_output_length, max_transition_length]
                links[b, i, j] represents the transition probability from the i-th vertex to **the (i+j)-th vertex**.
                (Note: this parameter is different from the torch version)
            output_length (torch.LongTensor):
                Shape: [batch_size]
                output_length should be the graph size, the vertices (index >= graph size) are ignored
            target_length (torch.LongTensor):
                Shape: [batch_size]
                target_length is the reference length, the tokens (index >= target length) are ignored

        Output (torch.LongTensor):
            Shape: [batch_size, max_output_length]
            if output[b, i]>=0, it represents the index of target token aligned with the i-th vertex
            otherwise, output[b, i] = -1, it represents the i-th vertex is not aligned with any target token
        """
        match_all = match_all.contiguous()
        links = links.contiguous()
        alpha, path = dag_loss_cuda.dag_best_alignment(
            match_all,
            links,
            output_length,
            target_length,
            DagBestAlignmentFunc.config,
        )  # bsz * prelen * tarlen
        path = path.to(torch.long)
        ctx.mark_non_differentiable(path)
        return path

    @staticmethod
    def backward(ctx, grad_output):
        assert False, "no backward function for best alignment"


dag_best_alignment = DagBestAlignmentFunc.apply


class DagLogsoftmaxGatherFunc(Function):

    @staticmethod
    def forward(ctx, word_ins_out, select_idx):  # bsz * prelen * vocabsize  # bsz * prelen * slen
        r"""
        This function is equivalent to the below codes:

            res = word_ins_out.log_softmax(dim=-1, dtype=torch.float).gather(-1, select_idx)

        Note: to reduce memory usage, word_ins_out is modified in place for storing backward tensors.
        DO NOT use word_ins_out after this function.
        If you do not like the side effect, please use the torch version instead

        Input:
            word_ins_out (torch.FloatTensor or torch.HalfTensor):
                Shape: [batch_size, max_output_length, vocab_size]
                the unnormalized logits
            select_idx (torch.LongTensor):
                Shape: [batch_size, max_output_length, select_id_size]
                index in gather function

        Output:
            modified_word_ins_out (torch.FloatTensor or torch.HalfTensor):
                Shape: [batch_size, max_output_length, vocab_size]
                modified word_ins_out, do not use it

            selected_result (torch.FloatTensor):
                Shape: [batch_size, max_output_length, select_id_size]
        """
        require_gradient = ctx.needs_input_grad[0]
        selected_result = dag_loss_cuda.logsoftmax_gather(word_ins_out, select_idx, require_gradient)
        # Note: the cuda kernel will modify word_ins_out and then reuse it in backward
        ctx.mark_dirty(word_ins_out)
        ctx.set_materialize_grads(False)

        if require_gradient:
            ctx.save_for_backward(word_ins_out, select_idx)
            ctx.has_backward = False
        return word_ins_out, selected_result  # bsz * prelen * slen

    @staticmethod
    def backward(ctx, grad_word_ins_out, grad_output):
        if not ctx.needs_input_grad[0]:
            return None, None
        assert grad_word_ins_out is None, "Cannot reuse word_ins_out after logsoftmax_gather"
        if grad_output is None:
            return None, None

        assert not ctx.has_backward, "Cannot backward twice in logsoftmax_gather"
        ctx.has_backward = True

        grad_input, selected_idx = ctx.saved_tensors
        grad_input.mul_(grad_output.sum(-1, keepdim=True).neg_().to(grad_input.dtype))
        grad_input.scatter_add_(-1, selected_idx, grad_output.to(grad_input.dtype))

        return grad_input, None


dag_logsoftmax_gather_inplace = DagLogsoftmaxGatherFunc.apply

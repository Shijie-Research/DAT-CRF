import torch
from torch import Tensor, jit


@jit.script
def logsumexp_keepdim(x: Tensor, dim: int) -> Tensor:
    # Solving nan issue when x contains -inf
    # See https://github.com/pytorch/pytorch/issues/31829
    m, _ = x.max(dim=dim, keepdim=True)
    mask = m == -float("inf")
    m = m.detach()
    s = (x - m.masked_fill_(mask, 0)).exp_().sum(dim=dim, keepdim=True)
    return s.masked_fill_(mask, 1).log_() + m.masked_fill_(mask, -float("inf"))


@jit.script
def loop_function_noempty(last_f: Tensor, links: Tensor, match: Tensor) -> Tensor:
    f_next = logsumexp_keepdim(last_f + links, 1)  # batch * 1 * prelen
    f_next = f_next.transpose(1, 2) + match  # batch * prelen * 1
    return f_next


@jit.script
def loop_function_noempty_max(last_f: Tensor, links: Tensor, match: Tensor) -> Tensor:
    f_next = torch.max(last_f + links, dim=1)[0]  # batch * 1 * prelen
    f_next = f_next.unsqueeze(-1) + match  # batch * prelen * 1
    return f_next


def torch_dag_loss(match_all, links, output_length, target_length):
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
            links[b, i, j] represents the transition probability from the i-th vertex to **the j-th vertex**.
            (Note: this parameter is different from the cuda version)
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
    match_all = match_all.transpose(1, 2)
    batch_size, prelen, tarlen = match_all.shape
    assert links.shape[1] == links.shape[2], "links should be batch_size * prelen * prelen"

    f_arr = []
    f_init = torch.zeros(batch_size, prelen, 1, dtype=match_all.dtype, device=match_all.device).fill_(float("-inf"))
    f_init[:, 0, 0] = match_all[:, 0, 0]
    f_arr.append(f_init)

    match_all_chunk = torch.chunk(match_all, tarlen, -1)  # k * [batch * prelen * 1]

    for k in range(1, tarlen):
        f_now = loop_function_noempty(f_arr[-1], links, match_all_chunk[k])
        f_arr.append(f_now)

    loss_result = torch.cat(f_arr, -1)[range(batch_size), output_length - 1, target_length - 1]

    return loss_result


def torch_max_loss(match_all, links, output_length, target_length):
    match_all = match_all.transpose(1, 2)
    batch_size, prelen, tarlen = match_all.shape
    assert links.shape[1] == links.shape[2], "links should be batch_size * prelen * prelen"

    f_arr = []
    f_init = torch.zeros(batch_size, prelen, 1, dtype=match_all.dtype, device=match_all.device).fill_(float("-inf"))
    f_init[:, 0, 0] = match_all[:, 0, 0]
    f_arr.append(f_init)

    match_arr = torch.chunk(match_all, tarlen, -1)
    for i in range(1, tarlen):
        f_now = loop_function_noempty_max(f_arr[-1], links, match_arr[i])
        f_arr.append(f_now)

    alllogprob = torch.cat(f_arr, -1)[range(batch_size), output_length - 1, target_length - 1]

    return alllogprob


def torch_dag_best_alignment(match_all, links, output_length, target_length):
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
            links[b, i, j] represents the transition probability from the i-th vertex to **the j-th vertex**.
            (Note: this parameter is different from the cuda version)
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
    with torch.enable_grad():
        match_all.requires_grad_()
        alllogprob = torch_max_loss(match_all, links, output_length, target_length)
        matchgrad = torch.autograd.grad(alllogprob.sum(), [match_all])[0]  # batch * talen * prelen
    pathvalue, path = matchgrad.max(dim=1)
    path.masked_fill_(pathvalue < 0.5, -1)
    return path


def torch_dag_logsoftmax_gather_inplace(word_ins_out, select_idx):
    r"""Fused operation of log_softmax and gather"""
    logits = torch.log_softmax(word_ins_out, -1, dtype=torch.float32)
    match = logits.gather(dim=-1, index=select_idx)
    return word_ins_out, match

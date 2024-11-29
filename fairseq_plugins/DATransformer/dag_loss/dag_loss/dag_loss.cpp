// ##########################################################################
// Copyright (C) 2022 COAI @ Tsinghua University

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//         http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ###########################################################################

#include <torch/extension.h>
#include <tuple>

#include "dag_loss.h"
#include "utilities.h"

std::tuple<torch::Tensor, torch::Tensor> dag_loss(
    const torch::Tensor &match_all,
    const torch::Tensor &links,
    const torch::Tensor &output_length,
    const torch::Tensor &target_length,
    bool require_gradient,
    int config)
{
  CHECK_CUDA(match_all);  // bsz * tarlen * prelen
  CHECK_CUDA(links);   // bsz * prelen * translen
  CHECK_CUDA(output_length); // bsz
  CHECK_CUDA(target_length); // bsz

  TORCH_CHECK(match_all.dim() == 3, "match_all dim != 3");
  TORCH_CHECK(links.dim() == 3, "links dim != 3");
  TORCH_CHECK(output_length.dim() == 1, "output_length dim != 3");
  TORCH_CHECK(target_length.dim() == 1, "target_length dim != 3");

  auto bsz = match_all.size(0);
  auto prelen = match_all.size(2);
  TORCH_CHECK(links.size(0) == bsz && output_length.size(0) == bsz && target_length.size(0) == bsz, "batch size not match");
  TORCH_CHECK(links.size(1) == prelen, "prelen not match");
  TORCH_CHECK(output_length.scalar_type() == at::kLong && target_length.scalar_type() == at::kLong, "length should be long");

  auto res = dag_loss_op(match_all, links, output_length, target_length, require_gradient, config);

  return res;
}

std::tuple<torch::Tensor, torch::Tensor> dag_loss_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &alpha,
    const torch::Tensor &beta,
    const torch::Tensor &match_all,
    const torch::Tensor &links,
    const torch::Tensor &output_length,
    const torch::Tensor &target_length,
    int config1,
    int config2) {
  auto res = dag_loss_backward_op(grad_output, alpha, beta, match_all, links, output_length, target_length, config1, config2);
  return res;
}

std::tuple<torch::Tensor, torch::Tensor> dag_best_alignment(
    const torch::Tensor &match_all,
    const torch::Tensor &links,
    const torch::Tensor &output_length,
    const torch::Tensor &target_length,
    int config) {
  CHECK_CUDA(match_all);  // bsz * tarlen * prelen
  CHECK_CUDA(links);   // bsz * prelen * translen
  CHECK_CUDA(output_length); // bsz
  CHECK_CUDA(target_length); // bsz

  TORCH_CHECK(match_all.dim() == 3, "match_all dim != 3");
  TORCH_CHECK(links.dim() == 3, "links dim != 3");
  TORCH_CHECK(output_length.dim() == 1, "output_length dim != 3");
  TORCH_CHECK(target_length.dim() == 1, "target_length dim != 3");

  auto bsz = match_all.size(0);
  auto prelen = match_all.size(2);
  TORCH_CHECK(links.size(0) == bsz && output_length.size(0) == bsz && target_length.size(0) == bsz, "batch size not match");
  TORCH_CHECK(links.size(1) == prelen, "prelen not match");

  TORCH_CHECK(output_length.scalar_type() == at::kLong && target_length.scalar_type() == at::kLong, "length should be long");

  auto res = dag_best_alignment_op(match_all, links, output_length, target_length, config);

  return res;
}

torch::Tensor logsoftmax_gather(
    torch::Tensor word_ins_out,
    const torch::Tensor &select_idx,
    bool require_gradient)
{
	CHECK_CUDA(word_ins_out);  // bsz * prelen * vocabsize
	CHECK_CUDA(select_idx);  // bsz * prelen * slen

	TORCH_CHECK(word_ins_out.dim() == 3, "word_ins_out dim != 3");
	TORCH_CHECK(select_idx.dim() == 3, "select_idx dim != 3");

  auto bsz = word_ins_out.size(0);
  auto prelen = word_ins_out.size(1);
  TORCH_CHECK(select_idx.size(0) == bsz, "batch size not match");
  TORCH_CHECK(select_idx.size(1) == prelen, "prelen size not match");

  TORCH_CHECK(select_idx.scalar_type() == at::kLong, "select_idx should be long");
  TORCH_CHECK(word_ins_out.is_contiguous(), "word_ins_out is not contiguous");

  auto res = logsoftmax_gather_op(word_ins_out, select_idx, require_gradient);

  return res;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dag_loss", &dag_loss, "DAG Loss");
  m.def("dag_loss_backward", &dag_loss_backward, "DAG Loss Backward");
  m.def("dag_best_alignment", &dag_best_alignment, "DAG Best Alignment");
  m.def("logsoftmax_gather", &logsoftmax_gather, "logsoftmax + gather");
}

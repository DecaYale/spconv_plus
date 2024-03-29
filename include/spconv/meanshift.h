// Copyright 2019 Yan Yan
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MEANSHIFT_TORCH_OP_H_
#define MEANSHIFT_TORCH_OP_H_

#include <cuda_runtime_api.h>
#include <spconv/indice.h>
#include <spconv/reordering.h>
#include <torch/script.h>
#include <torch_utils.h>
#include <utility/timer.h>
#include <spconv/nms_functor.h>

#include <unordered_map> 
namespace spconv {
// torch.jit's doc says only support int64, so we need to convert to int32.

template <typename T, typename Index>
void _meanshift_find_clusters(tv::CPU& d, tv::TensorView<T> input_centers,
                  tv::TensorView<const T> output_clusters, 
                  int size, double threshold=1e-3) {

    int B = input_centers.dim(0);
    int N = input_centers.dim(1);
    int K = input_centers.dim(2);

    
    const T* centers = input_centers.data();
    T* out = output_clusters.data();
    for(int b =0; b<B; ++b){
        for(int n=0; n<N; ++n){
            if 
        }
    } 

    for (int i = 0; i < size; ++i) {
      buf = buffer.data() + i * numPlanes;
      out = outFeatures.data() + indices[i] * numPlanes;
      for (int j = 0; j < numPlanes; ++j){
        out[j] += buf[j];
      }
    }
  }
template <typename T>
torch::Tensor
meanshift_find_clusters(torch::Tensor input, double iteration=10, double bandwidth=1) {
    // input: BxNxK, the results of cluster centers 

  tv::check_torch_dtype<T>(input);
  auto resOptions =
      torch::TensorOptions().dtype(torch::kInt64).device(input.device());

  if (input.size(0) == 0){
      return torch::zeros({0}, resOptions);
  }
  
  torch::Tensor indices;
//   if (preMaxSize > 0){
//       auto numKeepedScores = scores.size(0);
//       preMaxSize = std::min(numKeepedScores, preMaxSize);
//       auto res = torch::topk(scores, preMaxSize);
//       indices = std::get<1>(res);
//       boxes = torch::index_select(boxes, 0, indices);
//   }else{
//       indices = std::get<1>(torch::sort(scores));
//       boxes = torch::index_select(boxes, 0, indices);
//   }

  auto keep = torch::zeros({boxes.size(0)}, resOptions);
  int64_t keepNum = 0;
  if (boxes.device().type() == torch::kCPU) {
    auto nmsFunctor = functor::NonMaxSupressionFunctor<tv::CPU, T, int64_t>();
    keepNum = nmsFunctor(tv::CPU(), tv::torch2tv<int64_t>(keep),
    tv::torch2tv<const T>(boxes), T(thresh), T(eps));
  }else{
    TV_ASSERT_RT_ERR(false, "not implemented");
  }
  if (postMaxSize <= 0){
    postMaxSize = keepNum;
  }
  // std::cout << keep << std::endl;
  keep = keep.slice(0, 0, std::min(keepNum, postMaxSize));
  if (preMaxSize > 0){
    return torch::index_select(indices, 0, keep);
  }
  return keep;
}

} // namespace spconv

#endif
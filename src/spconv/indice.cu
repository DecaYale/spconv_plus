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

#include <ATen/ATen.h>
#include <chrono>
#include <limits>
#include <spconv/indice.cu.h>
#include <spconv/indice.h>
#include <spconv/mp_helper.h>
#include <tensorview/helper_launch.h>
#include <tensorview/tensorview.h>
#include <type_traits>
#include <utility/timer.h>

namespace spconv {
namespace functor {
template <typename Index, typename IndexGrid, unsigned NDim>
struct CreateConvIndicePairFunctorP1<tv::GPU, Index, IndexGrid, NDim> {
  Index operator()(const tv::GPU &d, tv::TensorView<const Index> indicesIn,
                   tv::TensorView<Index> indicesOut,
                   tv::TensorView<IndexGrid> gridsOut,
                   tv::TensorView<Index> indicePairs,
                   tv::TensorView<Index> indiceNum,
                   tv::TensorView<Index> indicePairUnique,
                   const tv::SimpleVector<Index, NDim> kernelSize,
                   const tv::SimpleVector<Index, NDim> stride,
                   const tv::SimpleVector<Index, NDim> padding,
                   const tv::SimpleVector<Index, NDim> dilation,
                   const tv::SimpleVector<Index, NDim> outSpatialShape,
                   bool transpose) {
    Index batchSize = gridsOut.dim(0);
    auto numActIn = indicesIn.dim(0);
    if (numActIn == 0)
      return 0;
    // auto timer = spconv::CudaContextTimer<>();
    if (transpose)
      prepareDeConvIndicePairsKernel<Index, IndexGrid, NDim, 4096>
          <<<tv::launch::getBlocks(numActIn), tv::launch::CUDA_NUM_THREADS, 0,
             d.getStream()>>>(indicesIn, indicesOut, gridsOut, indicePairs,
                              indiceNum, indicePairUnique, kernelSize, stride,
                              padding, dilation, outSpatialShape);
    else
      prepareIndicePairsKernel<Index, IndexGrid, NDim, 4096>
          <<<tv::launch::getBlocks(numActIn), tv::launch::CUDA_NUM_THREADS, 0,
             d.getStream()>>>(indicesIn, indicesOut, gridsOut, indicePairs,
                              indiceNum, indicePairUnique, kernelSize, stride,
                              padding, dilation, outSpatialShape);
    TV_CHECK_CUDA_ERR();
    // std::cout << "p1 gene time " << timer.report() / 1000.0 << std::endl;
    return 1;
  }
};

template <typename Index, typename IndexGrid, unsigned NDim>
struct CreateConvIndicePairFunctorP2<tv::GPU, Index, IndexGrid, NDim> {
  Index operator()(const tv::GPU &d, tv::TensorView<const Index> indicesIn,
                   tv::TensorView<Index> indicesOut,
                   tv::TensorView<IndexGrid> gridsOut,
                   tv::TensorView<Index> indicePairs,
                   tv::TensorView<Index> indiceNum,
                   tv::TensorView<Index> indicePairUnique,
                   const tv::SimpleVector<Index, NDim> outSpatialShape,
                   bool transpose, bool resetGrid) {
    Index batchSize = gridsOut.dim(0);
    auto kernelVolume = indicePairs.dim(0);
    auto numActIn = indicesIn.dim(0);
    if (numActIn == 0)
      return 0;
    Index numAct = indicePairUnique.dim(0) - 1;
    assignGridAndIndiceOutKernel<Index, IndexGrid, NDim>
        <<<tv::launch::getBlocks(numAct), tv::launch::CUDA_NUM_THREADS, 0,
           d.getStream()>>>(indicesOut, gridsOut, numAct, indicePairs,
                            indicePairUnique, outSpatialShape, batchSize);
    TV_CHECK_CUDA_ERR();
    assignIndicePairsKernel<Index, IndexGrid, NDim>
        <<<tv::launch::getBlocks(numActIn), tv::launch::CUDA_NUM_THREADS, 0,
           d.getStream()>>>(indicesOut, gridsOut, numActIn, indicePairs,
                            indicePairUnique, outSpatialShape);
    TV_CHECK_CUDA_ERR();
    if (resetGrid) {
      resetGridKernel<Index, IndexGrid, NDim>
          <<<tv::launch::getBlocks(numAct), tv::launch::CUDA_NUM_THREADS, 0,
             d.getStream()>>>(indicePairUnique.data(), gridsOut, numAct);
      TV_CHECK_CUDA_ERR();
    }
    return numAct;
  }
};

template <typename Index, typename IndexGrid, unsigned NDim>
struct CreateSubMIndicePairFunctor<tv::GPU, Index, IndexGrid, NDim> {
  Index operator()(const tv::GPU &d, tv::TensorView<const Index> indicesIn,
                   tv::TensorView<IndexGrid> gridsOut,
                   tv::TensorView<Index> indicePairs,
                   tv::TensorView<Index> indiceNum,
                   const tv::SimpleVector<Index, NDim> kernelSize,
                   const tv::SimpleVector<Index, NDim> stride,
                   const tv::SimpleVector<Index, NDim> padding,
                   const tv::SimpleVector<Index, NDim> dilation,
                   const tv::SimpleVector<Index, NDim> outSpatialShape,
                   bool transpose, bool resetGrid) {
    auto numActIn = indicesIn.dim(0);
    if (numActIn == 0)
      return 0;
    // auto timer = spconv::CudaContextTimer<>();
    prepareSubMGridKernel<Index, IndexGrid, NDim>
        <<<tv::launch::getBlocks(numActIn), tv::launch::CUDA_NUM_THREADS, 0,
           d.getStream()>>>(indicesIn, gridsOut, outSpatialShape);
    TV_CHECK_CUDA_ERR();
    getSubMIndicePairsKernel<Index, IndexGrid, NDim, 4096>
        <<<tv::launch::getBlocks(numActIn), tv::launch::CUDA_NUM_THREADS, 0,
           d.getStream()>>>(indicesIn, gridsOut, indicePairs, indiceNum,
                            kernelSize, stride, padding, dilation,
                            outSpatialShape);
    TV_CHECK_CUDA_ERR();
    // std::cout << "subm gene time " << timer.report() / 1000.0 << std::endl;
    if (resetGrid) {
      resetGridSubMKernel<Index, IndexGrid, NDim>
          <<<tv::launch::getBlocks(numActIn), tv::launch::CUDA_NUM_THREADS, 0,
             d.getStream()>>>(indicesIn.data(), gridsOut, outSpatialShape,
                              numActIn);
      TV_CHECK_CUDA_ERR();
    }
    return numActIn;
  }
};

//!!DY!!
template <typename Index, typename IndexGrid, unsigned NDim>
struct CreateConcatIndicePairFunctor<tv::GPU, Index, IndexGrid, NDim> {
  Index operator()(const tv::GPU &d, tv::TensorView<const Index> indicesIn1,
                   tv::TensorView<const Index> indicesIn2,
                   tv::TensorView<IndexGrid> gridsOut,
                   tv::TensorView<Index> indicePairs,
                   tv::TensorView<Index> indiceNum,
                   //  const tv::SimpleVector<Index, NDim> kernelSize,
                   //  const tv::SimpleVector<Index, NDim> stride,
                   //  const tv::SimpleVector<Index, NDim> padding,
                   //  const tv::SimpleVector<Index, NDim> dilation,
                   const tv::SimpleVector<Index, NDim> outSpatialShape,
                   tv::TensorView<Index> outInds, bool resetGrid) {

    auto numActIn1 = indicesIn1.dim(0);
    auto numActIn2 = indicesIn2.dim(0);

    if (numActIn1 == 0 && numActIn2 == 0)
      return 0;
    // auto timer = spconv::CudaContextTimer<>();
    prepareSubMGridKernel<Index, IndexGrid, NDim>
        <<<tv::launch::getBlocks(numActIn2), tv::launch::CUDA_NUM_THREADS, 0,
           d.getStream()>>>(indicesIn2, gridsOut, outSpatialShape);
    TV_CHECK_CUDA_ERR();

    // getSubMIndicePairsKernel<Index, IndexGrid, NDim, 4096>
    //     <<<tv::launch::getBlocks(numActIn), tv::launch::CUDA_NUM_THREADS, 0,
    //        d.getStream()>>>(indicesIn, gridsOut, indicePairs, indiceNum,
    //                      kernelSize, stride, padding, dilation,
    //                      outSpatialShape);
    getConcatIndicePairsKernel_step1<Index, IndexGrid, NDim, 4096>
        <<<tv::launch::getBlocks(numActIn1), tv::launch::CUDA_NUM_THREADS, 0,
           d.getStream()>>>(indicesIn1, indicesIn2, gridsOut, indicePairs,
                            indiceNum, outSpatialShape, outInds);
    TV_CHECK_CUDA_ERR();

    getConcatIndicePairsKernel_step2<Index, IndexGrid, NDim, 4096>
        <<<tv::launch::getBlocks(numActIn2), tv::launch::CUDA_NUM_THREADS, 0,
           d.getStream()>>>(indicesIn1, indicesIn2, gridsOut, indicePairs,
                            indiceNum, outSpatialShape, outInds);
    TV_CHECK_CUDA_ERR();

    // std::cout << "subm gene time " << timer.report() / 1000.0 << std::endl;
    if (resetGrid) {
      resetGridSubMKernel<Index, IndexGrid, NDim>
          <<<tv::launch::getBlocks(numActIn2), tv::launch::CUDA_NUM_THREADS, 0,
             d.getStream()>>>(indicesIn2.data(), gridsOut, outSpatialShape,
                              numActIn2);
      TV_CHECK_CUDA_ERR();
    }

    Index indiceNum_h[1];
    cudaMemcpy(indiceNum_h, (Index*)indiceNum.data(), 1*sizeof(Index), cudaMemcpyDeviceToHost);
    TV_CHECK_CUDA_ERR();
    // gpuTransferAddressToHost(inum,(int*)indiceNum.data(),  sizeof(int)); 
    
    // return numActIn;
    // return indiceNum(0);
    return indiceNum_h[0];
  }
};

} // namespace functor

#define DECLARE_GPU_SPECS_INDEX_NDIM(Index, NDIM)                              \
  template struct functor::CreateConvIndicePairFunctor<tv::GPU, Index, int,    \
                                                       NDIM>;                  \
  template struct functor::CreateConvIndicePairFunctorP1<tv::GPU, Index, int,  \
                                                         NDIM>;                \
  template struct functor::CreateConvIndicePairFunctorP2<tv::GPU, Index, int,  \
                                                         NDIM>;                \
  template struct functor::CreateSubMIndicePairFunctor<tv::GPU, Index, int,    \
                                                       NDIM>;                  \
  template struct functor::CreateConcatIndicePairFunctor<tv::GPU, Index, int,    \
                                                       NDIM>;

#define DECLARE_GPU_INDEX(Index)                                               \
  DECLARE_GPU_SPECS_INDEX_NDIM(Index, 1);                                      \
  DECLARE_GPU_SPECS_INDEX_NDIM(Index, 2);                                      \
  DECLARE_GPU_SPECS_INDEX_NDIM(Index, 3);                                      \
  DECLARE_GPU_SPECS_INDEX_NDIM(Index, 4);

DECLARE_GPU_INDEX(int);

#undef DECLARE_GPU_INDEX
#undef DECLARE_GPU_SPECS_INDEX_NDIM
} // namespace spconv
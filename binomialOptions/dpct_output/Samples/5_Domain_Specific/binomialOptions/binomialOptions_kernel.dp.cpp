/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

////////////////////////////////////////////////////////////////////////////////
// Global types and parameters
////////////////////////////////////////////////////////////////////////////////
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>

#include <helper_cuda.h>
#include "binomialOptions_common.h"
#include "realtype.h"
#include <cmath>

// Preprocessed input option data
typedef struct dpct_type_100429 {
  real S;
  real X;
  real vDt;
  real puByDf;
  real pdByDf;
} __TOptionData;
static dpct::constant_memory<__TOptionData, 1> d_OptionData(MAX_OPTIONS);
static dpct::global_memory<real, 1> d_CallValue(MAX_OPTIONS);

////////////////////////////////////////////////////////////////////////////////
// Overloaded shortcut functions for different precision modes
////////////////////////////////////////////////////////////////////////////////
#ifndef DOUBLE_PRECISION
inline float expiryCallValue(float S, float X, float vDt, int i) {
  float d = S * sycl::native::exp(vDt * (2.0f * i - NUM_STEPS)) - X;
  return (d > 0.0F) ? d : 0.0F;
}
#else
__device__ inline double expiryCallValue(double S, double X, double vDt,
                                         int i) {
  double d = S * exp(vDt * (2.0 * i - NUM_STEPS)) - X;
  return (d > 0.0) ? d : 0.0;
}
#endif

////////////////////////////////////////////////////////////////////////////////
// GPU kernel
////////////////////////////////////////////////////////////////////////////////
#define THREADBLOCK_SIZE 128
#define ELEMS_PER_THREAD (NUM_STEPS / THREADBLOCK_SIZE)
#if NUM_STEPS % THREADBLOCK_SIZE
#error Bad constants
#endif

/*
DPCT1110:0: The total declared local variable size in device function
binomialOptionsKernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void binomialOptionsKernel(const sycl::nd_item<3> &item_ct1,
                           __TOptionData const *d_OptionData, real *d_CallValue,
                           real *call_exchange) {
  // Handle to thread block group
  sycl::group<3> cta = item_ct1.get_group();

  const int tid = item_ct1.get_local_id(2);
  const real S = d_OptionData[item_ct1.get_group(2)].S;
  const real X = d_OptionData[item_ct1.get_group(2)].X;
  const real vDt = d_OptionData[item_ct1.get_group(2)].vDt;
  const real puByDf = d_OptionData[item_ct1.get_group(2)].puByDf;
  const real pdByDf = d_OptionData[item_ct1.get_group(2)].pdByDf;

  real call[ELEMS_PER_THREAD + 1];
#pragma unroll
  for (int i = 0; i < ELEMS_PER_THREAD; ++i)
    call[i] = expiryCallValue(S, X, vDt, tid * ELEMS_PER_THREAD + i);

  if (tid == 0)
    call_exchange[THREADBLOCK_SIZE] = expiryCallValue(S, X, vDt, NUM_STEPS);

  int final_it = sycl::max(0, tid * ELEMS_PER_THREAD - 1);

#pragma unroll 16
  for (int i = NUM_STEPS; i > 0; --i) {
    call_exchange[tid] = call[0];
    /*
    DPCT1065:1: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    call[ELEMS_PER_THREAD] = call_exchange[tid + 1];
    /*
    DPCT1065:2: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if (i > final_it) {
#pragma unroll
      for (int j = 0; j < ELEMS_PER_THREAD; ++j)
        call[j] = puByDf * call[j + 1] + pdByDf * call[j];
    }
  }

  if (tid == 0) {
    d_CallValue[item_ct1.get_group(2)] = call[0];
  }
}

////////////////////////////////////////////////////////////////////////////////
// Host-side interface to GPU binomialOptions
////////////////////////////////////////////////////////////////////////////////
extern "C" void binomialOptionsGPU(real *callValue, TOptionData *optionData,
                                   int optN) {
  __TOptionData h_OptionData[MAX_OPTIONS];

  for (int i = 0; i < optN; i++) {
    const real T = optionData[i].T;
    const real R = optionData[i].R;
    const real V = optionData[i].V;

    const real dt = T / (real)NUM_STEPS;
    const real vDt = V * sqrt(dt);
    const real rDt = R * dt;
    // Per-step interest and discount factors
    const real If = exp(rDt);
    const real Df = exp(-rDt);
    // Values and pseudoprobabilities of upward and downward moves
    const real u = exp(vDt);
    const real d = exp(-vDt);
    const real pu = (If - d) / (u - d);
    const real pd = (real)1.0 - pu;
    const real puByDf = pu * Df;
    const real pdByDf = pd * Df;

    h_OptionData[i].S = (real)optionData[i].S;
    h_OptionData[i].X = (real)optionData[i].X;
    h_OptionData[i].vDt = (real)vDt;
    h_OptionData[i].puByDf = (real)puByDf;
    h_OptionData[i].pdByDf = (real)pdByDf;
  }

  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(d_OptionData.get_ptr(), h_OptionData,
                                   optN * sizeof(__TOptionData))
                           .wait()));
  {
    d_OptionData.init();
    d_CallValue.init();

    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
      auto d_OptionData_ptr_ct1 = d_OptionData.get_ptr();
      auto d_CallValue_ptr_ct1 = d_CallValue.get_ptr();

      /*
      DPCT1101:17: 'THREADBLOCK_SIZE + 1' expression was replaced with a
      value. Modify the code to use the original expression, provided in
      comments, if it is correct.
      */
      sycl::local_accessor<real, 1> call_exchange_acc_ct1(
          sycl::range<1>(129 /*THREADBLOCK_SIZE + 1*/), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, optN) *
                                sycl::range<3>(1, 1, THREADBLOCK_SIZE),
                            sycl::range<3>(1, 1, THREADBLOCK_SIZE)),
          [=](sycl::nd_item<3> item_ct1) {
            binomialOptionsKernel(
                item_ct1, d_OptionData_ptr_ct1, d_CallValue_ptr_ct1,
                call_exchange_acc_ct1
                    .get_multi_ptr<sycl::access::decorated::no>()
                    .get());
          });
    });
  }
  getLastCudaError("binomialOptionsKernel() execution failed.\n");
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memcpy(callValue, d_CallValue.get_ptr(), optN * sizeof(real))
          .wait()));
}

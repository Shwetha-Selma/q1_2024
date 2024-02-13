#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
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
// Polynomial approximation of cumulative normal distribution function
////////////////////////////////////////////////////////////////////////////////
inline float cndGPU(float d) {
  const float A1 = 0.31938153f;
  const float A2 = -0.356563782f;
  const float A3 = 1.781477937f;
  const float A4 = -1.821255978f;
  const float A5 = 1.330274429f;
  const float RSQRT2PI = 0.39894228040143267793994605993438f;

  float K = 1.0f / (1.0f + 0.2316419f * sycl::fabs(d));

  float cnd = RSQRT2PI * sycl::native::exp(-0.5f * d * d) *
              (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

  if (d > 0) cnd = 1.0f - cnd;

  return cnd;
}

////////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
////////////////////////////////////////////////////////////////////////////////
inline void BlackScholesBodyGPU(float &CallResult, float &PutResult,
                                           float S,  // Stock price
                                           float X,  // Option strike
                                           float T,  // Option years
                                           float R,  // Riskless rate
                                           float V  // Volatility rate
                                           ) {
  float sqrtT, expRT;
  float d1, d2, CNDD1, CNDD2;

  sqrtT = 1.0F / sycl::rsqrt(T);
  d1 = (sycl::log(S / X) + (R + 0.5f * V * V) * T) / (V * sqrtT);
  d2 = d1 - V * sqrtT;

  CNDD1 = cndGPU(d1);
  CNDD2 = cndGPU(d2);

  // Calculate Call and Put simultaneously
  expRT = sycl::native::exp(-R * T);
  CallResult = S * CNDD1 - X * expRT * CNDD2;
  PutResult = X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1);
}

////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options on GPU
////////////////////////////////////////////////////////////////////////////////

    void BlackScholesGPU(sycl::float2 *__restrict d_CallResult,
                         sycl::float2 *__restrict d_PutResult,
                         sycl::float2 *__restrict d_StockPrice,
                         sycl::float2 *__restrict d_OptionStrike,
                         sycl::float2 *__restrict d_OptionYears, float Riskfree,
                         float Volatility, int optN,
                         const sycl::nd_item<3> &item_ct1) {
  ////Thread index
  // const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
  ////Total number of threads in execution grid
  // const int THREAD_N = blockDim.x * gridDim.x;

  const int opt = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

  // Calculating 2 options per thread to increase ILP (instruction level
  // parallelism)
  if (opt < (optN / 2)) {
    float callResult1, callResult2;
    float putResult1, putResult2;
    BlackScholesBodyGPU(callResult1, putResult1, d_StockPrice[opt].x(),
                        d_OptionStrike[opt].x(), d_OptionYears[opt].x(),
                        Riskfree, Volatility);
    BlackScholesBodyGPU(callResult2, putResult2, d_StockPrice[opt].y(),
                        d_OptionStrike[opt].y(), d_OptionYears[opt].y(),
                        Riskfree, Volatility);
    d_CallResult[opt] = sycl::float2(callResult1, callResult2);
    d_PutResult[opt] = sycl::float2(putResult1, putResult2);
  }
}

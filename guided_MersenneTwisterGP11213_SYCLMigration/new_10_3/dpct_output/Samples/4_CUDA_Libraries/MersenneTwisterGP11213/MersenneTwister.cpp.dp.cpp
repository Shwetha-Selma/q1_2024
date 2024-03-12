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

/*
 * This sample demonstrates the use of CURAND to generate
 * random numbers on GPU and CPU.
 */

// Utilities and system includes
// includes, system
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <dpct/rng_utils.hpp>

// Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cmath>

float compareResults(int rand_n, float *h_RandGPU, float *h_RandCPU);

const int DEFAULT_RAND_N = 2400000;
const unsigned int DEFAULT_SEED = 777;

///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  // Start logs
  printf("%s Starting...\n\n", argv[0]);

  // initialize the GPU, either identified by --device
  // or by picking the device with highest flop rate.
  int devID = findCudaDevice(argc, (const char **)argv);

  // parsing the number of random numbers to generate
  int rand_n = DEFAULT_RAND_N;

  if (checkCmdLineFlag(argc, (const char **)argv, "count")) {
    rand_n = getCmdLineArgumentInt(argc, (const char **)argv, "count");
  }

  printf("Allocating data for %i samples...\n", rand_n);

  // parsing the seed
  int seed = DEFAULT_SEED;

  if (checkCmdLineFlag(argc, (const char **)argv, "seed")) {
    seed = getCmdLineArgumentInt(argc, (const char **)argv, "seed");
  }

  printf("Seeding with %i ...\n", seed);

  dpct::queue_ptr stream;
  /*
  DPCT1025:14: The SYCL queue is created ignoring the flag and priority options.
  */
  checkCudaErrors(
      DPCT_CHECK_ERROR(stream = dpct::get_current_device().create_queue()));

  float *d_Rand;
  checkCudaErrors(DPCT_CHECK_ERROR(
      d_Rand = sycl::malloc_device<float>(rand_n, dpct::get_in_order_queue())));

  dpct::rng::host_rng_ptr prngGPU;
  checkCudaErrors(DPCT_CHECK_ERROR(prngGPU = dpct::rng::create_host_rng(
                                       dpct::rng::random_engine_type::mt2203)));
  checkCudaErrors(DPCT_CHECK_ERROR(prngGPU->set_queue(stream)));
  checkCudaErrors(DPCT_CHECK_ERROR(prngGPU->set_seed(seed)));

  dpct::rng::host_rng_ptr prngCPU;
  checkCudaErrors(DPCT_CHECK_ERROR(prngCPU = dpct::rng::create_host_rng(
                                       dpct::rng::random_engine_type::mt2203,
                                       dpct::cpu_device().default_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(prngCPU->set_seed(seed)));

  //
  // Example 1: Compare random numbers generated on GPU and CPU
  float *h_RandGPU;
  checkCudaErrors(DPCT_CHECK_ERROR(h_RandGPU = sycl::malloc_host<float>(
                                       rand_n, dpct::get_in_order_queue())));

  printf("Generating random numbers on GPU...\n\n");
  checkCudaErrors(
      DPCT_CHECK_ERROR(prngGPU->generate_uniform((float *)d_Rand, rand_n)));

  printf("\nReading back the results...\n");
  checkCudaErrors(DPCT_CHECK_ERROR(
      stream->memcpy(h_RandGPU, d_Rand, rand_n * sizeof(float))));

  float *h_RandCPU = (float *)malloc(rand_n * sizeof(float));

  printf("Generating random numbers on CPU...\n\n");
  checkCudaErrors(
      DPCT_CHECK_ERROR(prngCPU->generate_uniform((float *)h_RandCPU, rand_n)));

  checkCudaErrors(DPCT_CHECK_ERROR(stream->wait()));
  printf("Comparing CPU/GPU random numbers...\n\n");
  float L1norm = compareResults(rand_n, h_RandGPU, h_RandCPU);

  //
  // Example 2: Timing of random number generation on GPU
  const int numIterations = 10;
  int i;
  StopWatchInterface *hTimer;

  sdkCreateTimer(&hTimer);
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);

  for (i = 0; i < numIterations; i++) {
    checkCudaErrors(
        DPCT_CHECK_ERROR(prngGPU->generate_uniform((float *)d_Rand, rand_n)));
  }

  checkCudaErrors(DPCT_CHECK_ERROR(stream->wait()));
  sdkStopTimer(&hTimer);

  double gpuTime = 1.0e-3 * sdkGetTimerValue(&hTimer) / (double)numIterations;

  printf(
      "MersenneTwisterGP11213, Throughput = %.4f GNumbers/s, Time = %.5f s, "
      "Size = %u Numbers\n",
      1.0e-9 * rand_n / gpuTime, gpuTime, rand_n);

  printf("Shutting down...\n");

  checkCudaErrors(DPCT_CHECK_ERROR(prngGPU.reset()));
  checkCudaErrors(DPCT_CHECK_ERROR(prngCPU.reset()));
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::get_current_device().destroy_queue(stream)));
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::dpct_free(d_Rand, dpct::get_in_order_queue())));
  sdkDeleteTimer(&hTimer);
  checkCudaErrors(
      DPCT_CHECK_ERROR(sycl::free(h_RandGPU, dpct::get_in_order_queue())));
  free(h_RandCPU);

  exit(L1norm < 1e-6 ? EXIT_SUCCESS : EXIT_FAILURE);
}

float compareResults(int rand_n, float *h_RandGPU, float *h_RandCPU) {
  int i;
  float rCPU, rGPU, delta;
  float max_delta = 0.;
  float sum_delta = 0.;
  float sum_ref = 0.;

  for (i = 0; i < rand_n; i++) {
    rCPU = h_RandCPU[i];
    rGPU = h_RandGPU[i];
    delta = fabs(rCPU - rGPU);
    sum_delta += delta;
    sum_ref += fabs(rCPU);

    if (delta >= max_delta) {
      max_delta = delta;
    }
  }

  float L1norm = (float)(sum_delta / sum_ref);
  printf("Max absolute error: %E\n", max_delta);
  printf("L1 norm: %E\n\n", L1norm);

  return L1norm;
}

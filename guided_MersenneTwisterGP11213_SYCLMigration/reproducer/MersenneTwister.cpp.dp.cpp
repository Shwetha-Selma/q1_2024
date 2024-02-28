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


const int DEFAULT_RAND_N = 10240;//2400000;
const unsigned int DEFAULT_SEED = 777;

///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  // Start logs
  printf("%s Starting...\n\n", argv[0]);

  // parsing the number of random numbers to generate
  int rand_n = DEFAULT_RAND_N;

  if (checkCmdLineFlag(argc, (const char **)argv, "count")) {
    rand_n = getCmdLineArgumentInt(argc, (const char **)argv, "count");
  }

  printf("Allocating data for %i samples...\n", rand_n);
    
  sycl::queue q{sycl::default_selector_v};

  // parsing the seed
  int seed = DEFAULT_SEED;

  if (checkCmdLineFlag(argc, (const char **)argv, "seed")) {
    seed = getCmdLineArgumentInt(argc, (const char **)argv, "seed");
  }

  printf("Seeding with %i ...\n", seed);

      dpct::queue_ptr stream;
  DPCT_CHECK_ERROR(stream = dpct::get_current_device().create_queue());

  float *d_Rand;
  DPCT_CHECK_ERROR(
      d_Rand = sycl::malloc_device<float>(rand_n, dpct::get_in_order_queue()));

  dpct::rng::host_rng_ptr prngGPU;
    
  DPCT_CHECK_ERROR(prngGPU = dpct::rng::create_host_rng(
                                        dpct::rng::random_engine_type::mt2203));
  DPCT_CHECK_ERROR(prngGPU->set_queue(stream));
  DPCT_CHECK_ERROR(prngGPU->set_seed(seed));
  DPCT_CHECK_ERROR(prngGPU->set_engine_idx(1));

  dpct::rng::host_rng_ptr prngCPU;
  prngCPU = dpct::rng::create_host_rng<true>(
                                       dpct::rng::random_engine_type::mt2203);
  DPCT_CHECK_ERROR(prngCPU->set_seed(seed));
  DPCT_CHECK_ERROR(prngCPU->set_engine_idx(1));

  //
  // Example 1: Compare random numbers generated on GPU and CPU
  float *h_RandGPU;
  DPCT_CHECK_ERROR(h_RandGPU = sycl::malloc_host<float>(
                                       rand_n, dpct::get_in_order_queue()));

  printf("Generating random numbers on GPU...\n\n");
  DPCT_CHECK_ERROR(prngGPU->generate_uniform((float *)d_Rand, rand_n));

  printf("\nReading back the results...\n");
  DPCT_CHECK_ERROR(stream->memcpy(h_RandGPU, d_Rand, rand_n * sizeof(float)));


  float *h_RandCPU;
  DPCT_CHECK_ERROR(h_RandCPU = sycl::malloc_host<float>(
                                      rand_n, dpct::get_in_order_queue()));
  printf("Generating random numbers on CPU...\n\n");
  DPCT_CHECK_ERROR(prngCPU->generate_uniform((float *)h_RandCPU, rand_n));


  checkCudaErrors(DPCT_CHECK_ERROR(stream->wait()));

    int count =0;

  for (int i = 0; i < rand_n; i++) {
        if (fabs(h_RandCPU[i] - h_RandGPU[i]) > 0.1f)
            printf("mismatch i=%d h_RandGPU=%f h_RandCPU=%f\n", i, h_RandGPU[i],
                h_RandCPU[i]);
        
        else
            count++;
    }
    if (count == rand_n)
        printf("TEST PASSED\n");
    else
        printf("count value: %d\t TEST FAILED\n", count);



  printf("Shutting down...\n");

  DPCT_CHECK_ERROR(prngGPU.reset());
  prngCPU.reset();
  DPCT_CHECK_ERROR(dpct::get_current_device().destroy_queue(stream));
  DPCT_CHECK_ERROR(sycl::free(d_Rand, dpct::get_in_order_queue()));
  DPCT_CHECK_ERROR(sycl::free(h_RandGPU, dpct::get_in_order_queue()));
  DPCT_CHECK_ERROR(sycl::free(h_RandCPU, dpct::get_in_order_queue()));


  return 0;
}



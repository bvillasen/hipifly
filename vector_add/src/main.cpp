
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "gpu_functions.h"

int main( int argc, char *argv[] ){

  printf( "GPU vector addition \n" );

  // Set the GPU device
  int device_id = 0;  
  set_device( device_id );

  
  const uint N = 256 * 256 * 256 * 32;
  printf( "N elements: %d \n", N );
  
  //  Allocate host arrays  
  double *h_a, *h_b, *h_c; 
  h_a = (double *) malloc( N*sizeof(double) );
  h_b = (double *) malloc( N*sizeof(double) );
  h_c = (double *) malloc( N*sizeof(double) );

  // Initialize host arrays
  for ( uint i=0; i<N; ++i ){
    h_a[i] = i;
    h_b[i] = i/4.f;
  }
  
  // Allocate device arrays and copy the data from the host 
  double *d_a, *d_b, *d_c;
  allocate_device_arrays( N, d_a, d_b, d_c );
  copy_host_to_device( N, h_a, h_b, d_a, d_b );

  // Perform the vector addition (c = a + b) on the GPU
  float kernel_time;
  kernel_time = gpu_vector_add( N, d_a, d_b, d_c );

  printf( "Kernel executed in %.2f milliseconds. \n", kernel_time);
  printf( "BW = %.1f GB/s. \n", 3*N*sizeof(double)/(kernel_time*1e-3)/(1024*1024*1024) );

  // Copy result from device to host
  copy_device_to_host( N, d_c, h_c );


  // Validate the results
  bool validation_passed = true;
  for ( int i=0; i<N; i++ ){
    if ( h_c[i] != ( h_a[i] + h_b[i] ) ){
      printf( "ERROR: Result doesn't match expected value: %f   %f \n", h_c[i], h_a[i] + h_b[i] );
      validation_passed = false;
    }
  }
  if (validation_passed ) printf( "Validation PASSED. \n");

  printf( "Finished \n" );

}

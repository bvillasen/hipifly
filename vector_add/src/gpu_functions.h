
#ifndef GPU_FUNCTIONS_H
#define GPU_FUNCTIONS_H

int set_device( int device_id );

void initialize_device_data( int N, double *&d_a, double *&d_b, double *&d_c  );

void allocate_device_arrays( int N, double *&d_a, double *&d_b, double *&d_c  );

void copy_host_to_device( int N, double *h_a, double *h_b, 
                          double *&d_a, double *&d_b   );

float gpu_vector_add( int N, double *d_a, double *d_b, double *d_c );                          

void copy_device_to_host( int N, double *d_a, double *h_a );

#endif
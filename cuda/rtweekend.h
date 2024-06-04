#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <curand_kernel.h> // Include CUDA random number generator

// C++ Std Usings

using std::make_shared;
using std::shared_ptr;
using std::sqrt;

// Constants

constexpr double inf = std::numeric_limits<double>::infinity();
constexpr double pi = 3.1415926535897932385;

// Utility Functions

__host__ __device__ inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}

__device__ inline double random_double(curandState* local_rand_state) {
    // Returns a random real in [0,1) using CUDA random number generator
    return curand_uniform(local_rand_state);
}

__global__ void init_rand_state(curandState* rand_state, unsigned long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &rand_state[id]);
}

// Common Headers
#include "interval.h"
#include "color.h"
#include "ray.h"
#include "vec3.h"

#endif

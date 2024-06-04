#include <iostream>
#include <fstream>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"
#include "color.h"
#include "rtweekend.h"
#include <cuda_runtime.h>

// ############################################################################
//                          KERNEL CODE
// ############################################################################

// Device function for color calculation
__device__ color ray_color(const ray& r, hittable** world) {
    hit_record rec;
    if ((*world)->hit(r, interval(0.0, inf), rec)) {
        vec3 N = rec.normal;
        return 0.5 * color(N.x() + 1, N.y() + 1, N.z() + 1);
    }
    vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

// Rendering the scene
__global__ void render(vec3* fb, int max_x, int max_y, int samples_per_pixel, camera cam, hittable** world, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    color pixel_color(0, 0, 0);
    for (int s = 0; s < samples_per_pixel; ++s) {
        auto u = double(i + random_double(&local_rand_state)) / (max_x - 1);
        auto v = double(j + random_double(&local_rand_state)) / (max_y - 1);
        ray r = cam.get_ray(u, v, &local_rand_state);
        pixel_color += ray_color(r, world);
    }
    rand_state[pixel_index] = local_rand_state;
    fb[pixel_index] = pixel_color / samples_per_pixel;
}

// Creating the world
__global__ void create_world(hittable** d_list, hittable** d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_list[0] = new sphere(point3(0, 0, -1), 0.5);
        d_list[1] = new sphere(point3(0, -100.5, -1), 100);
        *d_world = new hittable_list(d_list, 2);
    }
}

// Freeing the world
__global__ void free_world(hittable** d_list, hittable** d_world) {
    delete ((sphere*)d_list[0]);
    delete ((sphere*)d_list[1]);
    delete ((hittable_list*)*d_world);
}

// Initializing random states
__global__ void init_rand_state_kernel(curandState* rand_state, unsigned long long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &rand_state[idx]);
}

// ############################################################################
//                          MAIN CODE
// ############################################################################

int main() {
    // Initialize host variables ----------------------------------------------
    // Image
    const int image_width = 400;
    const int image_height = static_cast<int>(image_width / 16.0 * 9.0);
    const int samples_per_pixel = 100;
    
    // Allocate device variables ----------------------------------------------

    // Allocate FB
    int num_pixels = image_width * image_height;
    size_t fb_size = num_pixels * sizeof(vec3);
    vec3* fb;
    cudaMalloc((void**)&fb, fb_size);

    // Random state
    curandState* d_rand_state;
    cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState));

    // Initialize frame buffer
    cudaMemset(fb, 0, fb_size);

    // World
    hittable** d_list;
    cudaMalloc((void**)&d_list, 2 * sizeof(hittable*));
    hittable** d_world;
    cudaMalloc((void**)&d_world, sizeof(hittable*));
    create_world<<<1, 1>>>(d_list, d_world);
    cudaDeviceSynchronize();
    
    // Launch kernel ----------------------------------------------------------

    // Initialize random states
    init_rand_state_kernel<<<(num_pixels + 255) / 256, 256>>>(d_rand_state, time(0));
    cudaDeviceSynchronize();

    // Camera
    camera cam;

    // Timing variables
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start);
    
    // Initialize thread block and kernel grid dimensions ---------------------
    // Render buffer
    dim3 blocks((image_width + 15) / 16, (image_height + 15) / 16);
    dim3 threads(16, 16);

    // Invoke CUDA kernel -----------------------------------------------------

    render<<<blocks, threads>>>(fb, image_width, image_height, samples_per_pixel, cam, d_world, d_rand_state);
    cudaDeviceSynchronize();

    // Allocate host memory for the frame buffer
    vec3* host_fb = new vec3[num_pixels];

    // Copy device variables from host ----------------------------------------

    cudaMemcpy(host_fb, fb, fb_size, cudaMemcpyDeviceToHost);

    // Output FB as image
    std::ofstream output_file("output_cuda.ppm");
    output_file << "P3\n" << image_width << " " << image_height << "\n255\n";
    for (int j = image_height - 1; j >= 0; --j) {
        std::clog << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; ++i) {
            int pixel_index = j * image_width + i;
            write_color(output_file, host_fb[pixel_index]);
        }
    }
    output_file.close();

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "\nRender Time (CUDA): " << milliseconds << " milliseconds\n";

    // Free memory ------------------------------------------------------------
    delete[] host_fb;
    free_world<<<1, 1>>>(d_list, d_world);
    cudaDeviceSynchronize();
    cudaFree(d_world);
    cudaFree(d_list);
    cudaFree(fb);
    cudaFree(d_rand_state);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

#pragma once
#include <math.h>
#include "ray.h"
#include "rtweekend.h"
#include "hittable.h"
#include <curand_kernel.h>

class camera {
public:
    double aspect_ratio = 16.0 / 9.0; // Ratio of image width over height
    int    image_width  = 400; // Rendered image width in pixel count
    int    samples_per_pixel = 100; // Count of random samples for each pixel

    __host__ __device__ camera() {
        initialize();
    }

    __device__ ray get_ray(double u, double v, curandState* rand_state) const {
        return ray(origin, lower_left_corner + u * horizontal + v * vertical - origin);
    }

    __host__ __device__ void initialize() {
        auto viewport_height = 2.0;
        auto viewport_width = aspect_ratio * viewport_height;
        auto focal_length = 1.0;

        origin = point3(0, 0, 0);
        horizontal = vec3(viewport_width, 0.0, 0.0);
        vertical = vec3(0.0, viewport_height, 0.0);
        lower_left_corner = origin - horizontal / 2 - vertical / 2 - vec3(0, 0, focal_length);
    }

private:
    point3 origin;
    point3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
};

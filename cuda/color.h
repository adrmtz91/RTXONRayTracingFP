#ifndef COLOR_H
#define COLOR_H

#include "interval.h"
#include "vec3.h"

using color = vec3;

__host__ __device__ void clamp_color(const color &pixel_color, int &rbyte, int &gbyte, int &bbyte)
{
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

     // Translate the [0,1] component values to the byte range [0,255].
    interval intensity(0.000, 0.999);
    rbyte = int(256 * intensity.clamp(r));
    gbyte = int(256 * intensity.clamp(g));
    bbyte = int(256 * intensity.clamp(b));
}

void write_color(std::ostream &out, const color &pixel_color)
{
    int rbyte, gbyte, bbyte;
    clamp_color(pixel_color, rbyte, gbyte, bbyte);
    // Write out the pixel color components.
    out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}

#endif

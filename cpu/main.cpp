
#include "camera.h"
#include "rtweekend.h"
#include "hittable_list.h"
#include "sphere.h"
#include <chrono>
using namespace std;
using namespace std::chrono;

int main() {
    hittable_list world;

    world.add(make_shared<sphere>(point3(0,0,-1), 0.5));
    world.add(make_shared<sphere>(point3(0,-100.5,-1), 100));

    camera cam;

    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width  = 400;
    cam.samples_per_pixel = 100;
    
    auto start = high_resolution_clock::now();
    cam.render(world);
    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(stop - start);
    std::clog << "Render Time (C++): " << duration.count() << " milliseconds" << std::endl;

}
#pragma once
#include "hittable.h"

class hittable_list : public hittable {
public:
    hittable** objects;
    int object_count;

    __host__ __device__ hittable_list() : objects(nullptr), object_count(0) {}
    __host__ __device__ hittable_list(hittable** objs, int count) : objects(objs), object_count(count) {}

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_t.max;

        for (int i = 0; i < object_count; i++) {
            if (objects[i]->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }
};

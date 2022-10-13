#ifndef LERP_HH_
#define LERP_HH_

#include <vector>
#include <array>

namespace lerp {

    std::vector<std::tuple<std::array<double, 6>, std::array<int, 7>, std::array<double, 7>>> get_frames(size_t num_frames, const std::vector<std::array<double, 6>>& camera_data);
    
}

#endif

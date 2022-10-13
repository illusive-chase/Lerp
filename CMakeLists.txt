add_definitions(-DCGAL_EIGEN3_ENABLED)
find_package(CGAL REQUIRED)

include_directories(/usr/local/include/eigen3)

add_library(lerp_lib lerp.cc)
target_link_libraries(lerp_lib CGAL::CGAL)

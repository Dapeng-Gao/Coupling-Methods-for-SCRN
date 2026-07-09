cmake_minimum_required(VERSION 3.12)
project(StochasticCouplingSimulation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_definitions(NDEBUG EIGEN_NO_DEBUG)

find_package(Eigen3 REQUIRED)

if(APPLE)
    set(OpenMP_ROOT "/opt/homebrew/opt/libomp")
endif()
find_package(OpenMP REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(SOURCES
    CTMC_coupling.cpp
    CRN.cpp
    Dictionary.cpp
    Poisson.cpp
    Reflection_coupling.cpp
)

add_executable(coupling ${SOURCES})

target_link_libraries(coupling PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX)

if(NOT MSVC)
    target_compile_options(coupling PRIVATE -O3 -march=native -Wall)
endif()
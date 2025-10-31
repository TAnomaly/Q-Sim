cmake_minimum_required(VERSION 3.25)
project(QSim VERSION 2.0 LANGUAGES CXX)

# C++23 with maximum optimizations
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -flto -ffast-math")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -funroll-loops -fomit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG")

# Additional performance flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall -Wextra -Wpedantic
        -mavx2           # SIMD instructions
    )
endif()

# Find required packages
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)

# Main DQN executable
add_executable(qsim_dqn
    main_dqn.cpp
)

target_link_libraries(qsim_dqn PRIVATE
    OpenGL::GL
    glfw
    opencv_core
    opencv_imgproc
    opencv_highgui
    m  # Math library
    pthread  # Threading
)

# Enable interprocedural optimization (LTO)
set_property(TARGET qsim_dqn PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

# Profile-Guided Optimization (PGO) support
option(ENABLE_PGO_GENERATE "Generate PGO profile" OFF)
option(ENABLE_PGO_USE "Use PGO profile" OFF)

if(ENABLE_PGO_GENERATE)
    target_compile_options(qsim_dqn PRIVATE -fprofile-generate)
    target_link_options(qsim_dqn PRIVATE -fprofile-generate)
endif()

if(ENABLE_PGO_USE)
    target_compile_options(qsim_dqn PRIVATE -fprofile-use -fprofile-correction)
    target_link_options(qsim_dqn PRIVATE -fprofile-use)
endif()

# Print build configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ Standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "Optimization flags: ${CMAKE_CXX_FLAGS_RELEASE}")

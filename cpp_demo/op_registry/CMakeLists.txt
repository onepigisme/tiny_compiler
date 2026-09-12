cmake_minimum_required(VERSION 3.14)
project(tml_op_registry LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(tml_demo
    main.cpp
    src/layers.cpp
)

target_include_directories(tml_demo PRIVATE include)

if(MSVC)
    target_compile_options(tml_demo PRIVATE /W4 /utf-8)
else()
    target_compile_options(tml_demo PRIVATE -Wall -Wextra -Wpedantic)
endif()

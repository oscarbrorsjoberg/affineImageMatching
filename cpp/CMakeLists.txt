cmake_minimum_required(VERSION 3.17)
project("viewChange" VERSION 0.0.1)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV 4.7 REQUIRED)

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "Affine maps requires OpenCV!")
endif()

set(build_flags
    "-std=c++17"
    "-lstdc++fs" # filesystem
    # "Wall"
    # "Wextra"
    "-pedantic"
    "-O0"
    "-g"
    )

add_executable(${PROJECT_NAME}
    src/main.cpp
    )

target_compile_options(${PROJECT_NAME}
	PUBLIC
	${build_flags}
)

# target_include_directories(${PROJECT_NAME}
#   PUBLIC
#   ${OpenCV_INCLUDE_DIRS}
#     )

target_link_libraries(${PROJECT_NAME}
  PUBLIC
	${OpenCV_LIBS}
    )

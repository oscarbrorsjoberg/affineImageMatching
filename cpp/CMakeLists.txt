cmake_minimum_required(VERSION 3.17)
project("viewChange" VERSION 0.0.1)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

message("${CMAKE_MODULE_PATH}")

if("${EXIV2_INCLUDES}" STREQUAL "")
  message(FATAL_ERROR "Affine maps require Exiv2 -- please EXIV2_INCLUDES to include folder")
endif()

find_package(OpenCV 4.7 REQUIRED)
# find_package(Exiv2 REQUIRED)

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "Affine maps requires OpenCV!")
endif()

set(build_flags
    "-std=c++17"
    "-lstdc++fs" # filesystem
    "-pedantic"
    "-O0"
    "-g"
    )



## find_model
add_library(
    find_model
    utils/find_model.cpp

)

target_compile_options(find_model
  INTERFACE
  ${build_flags}
)

target_link_libraries(find_model
  PRIVATE
  ${OpenCV_LIBS}
)
target_include_directories(find_model
  PRIVATE
  utils/
)

## genIO
add_library(
    genIO
    utils/genIO.cpp
)

target_compile_options(genIO
  INTERFACE
  ${build_flags}
)

target_link_libraries(genIO
  PRIVATE
  ${OpenCV_LIBS}
)
target_include_directories(genIO
  PRIVATE
  utils/
)


## main
add_executable(${PROJECT_NAME}
    src/main.cpp
)

target_compile_options(${PROJECT_NAME}
  PRIVATE
  ${build_flags}
)

target_include_directories(${PROJECT_NAME}
  PRIVATE
  ${EXIV2_INCLUDES}
  utils/
)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
  ${OpenCV_LIBS}
  exiv2      
  find_model
  genIO
)

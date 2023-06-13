cmake_minimum_required(VERSION 3.17)
project("viewChange" VERSION 0.0.1)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

find_package(OpenCV 4.7 REQUIRED)
# find_package(Exiv2 REQUIRED)

message("${EXIV2_INCLUDE_DIR}")
message("${EXIV2_LIBRARIES}")

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
  ${EXIV2_INCLUDE_DIR}
  utils/
)

# target_link_directories(${PROJECT_NAME}
#   PRIVATE
#   ${EXIV2_LIBRARIES}
# )

target_link_libraries(${PROJECT_NAME}
  PRIVATE
  ${OpenCV_LIBS}
  # ${EXIV2_LIBRARY}
  # exiv2
  find_model
  genIO
)

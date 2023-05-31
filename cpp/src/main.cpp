#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>

static std::string exe_name(){
  std::string out;
  std::ifstream("/proc/self/comm") >> out;
  return out;
}

int main(int argc, char *argv[])
{
  try {
    if(argc < 3){
      throw std::runtime_error("Missing input parameters, " +
          exe_name() +
          " <im1> <im2>");
    }

  }
  catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}


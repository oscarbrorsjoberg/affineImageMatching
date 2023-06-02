#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <optional>
#include <typeinfo>

namespace fs = std::filesystem;

static std::string exe_name(){
  std::string out;
  std::ifstream("/proc/self/comm") >> out;
  return out;
}

class inputArgBase{
  protected:
    bool updated_;
  public:
    virtual ~inputArgBase() = default;

    inputArgBase(const std::string &name, const std::string &short_name, 
              const std::string &description, bool required = false):
      name_(name),
      short_name_(short_name),
      description_(description),
      required_(required),
      updated_(false)
    {
    }
    virtual void set_value_from_string(const std::string &inpt) = 0;
    virtual std::string value_str() = 0;
    virtual bool value_is_set() = 0;

    bool is_updated(){
      return updated_;
    }

    bool is_required(){
      return required_;
    }

    bool required_;
    std::string name_;
    std::string short_name_;
    std::string description_;
};

template <typename T>
class inputArg : public inputArgBase {

  private:
    T *value_;

  public:
    inputArg(const std::string &name, const std::string &short_name, 
              const std::string &description, T* default_value, bool required = false):
      inputArgBase(name, short_name, description, required),
      value_(default_value)
        {
        }

    T get_value(){return *value_;}
    void set_value(T value){*value_ = value;}
    bool value_is_set() {return value_ != nullptr;}
    
    std::string value_str(){
      std::ostringstream os;
      if(value_){
        if(os << *value_){
          return os.str();
        }
        else{
          throw std::runtime_error("Unable to set value!");
        }
      }
      else{
        return "Empty";
      }
    }

    void set_value_from_string(const std::string &inpt){
      std::istringstream iss(inpt);
      T value;
      if(iss >> value){
        set_value(value);
        updated_ = true;
      }
      else{
        throw std::runtime_error("Unable to set value!");
      }
    }


};



class appInputOpts {
  public:
    std::vector<std::unique_ptr<inputArgBase>> pargs;

    template<typename T>
    void add_argument(const std::string &long_name, const std::string &short_name,
                 const std::string &description, T *var, bool required){
      pargs.push_back(std::make_unique<inputArg<T>>( 
                     long_name, short_name, description, 
                     var, required));
    }

    void help(){
      std::cout << " This is " << exe_name() << " a simple copy of OpenCVs asift. " << std::endl;
      std::cout << " Input parameters:" << std::endl;
      for(auto &parg: pargs){
        std::cout << "\t" << parg->name_ << " " << parg->short_name_ << " | " << parg->description_  << " | " ;
        std::cout << parg->value_str() << "\n";
      }
    }

    bool parse_args(int argc, char *args[]){
      if(argc == 1 || ((argc - 1) % 2) != 0){
        return false;
      }
      for(int i = 1; i < argc; i+=2){
        std::string current_arg_key{args[i]};
        std::string current_arg_value{args[i + 1]};
        for(auto &parg: pargs){
          if(current_arg_key == parg->name_ || current_arg_key == parg->short_name_){
            parg->set_value_from_string(current_arg_value);
          }
        }
      }

      bool out = true;
      for(auto &parg: pargs)
        out &= parg->is_required() ? parg->is_updated(): true;

      return out;
    }
};

int main(int argc, char *argv[])
{
  try {
    std::string im0_path, im1_path, 
      kpt_type = "orb";
    bool use_flann = false;

    appInputOpts opts;
    opts.add_argument("--image0", "-im0", "path to first image", &im0_path, true);
    opts.add_argument("--image1", "-im1", "path to second image", &im1_path, true);
    opts.add_argument("--keypoint", "-k", "kpt type", &kpt_type, false);
    opts.add_argument("--flann", "-f",    "if using flann or brute force matching", &use_flann, false);

    if(!opts.parse_args(argc, argv)){
      opts.help();
      throw std::runtime_error("Non-valid input");
    }

    assert(fs::exists(fs::path(im0_path)));
    assert(fs::exists(fs::path(im1_path)));

    cv::Mat im0 = cv::imread(im0_path, cv::IMREAD_GRAYSCALE);
    cv::Mat im1 = cv::imread(im1_path, cv::IMREAD_GRAYSCALE);

    /* kpt_type.lower_case(); */

    std::transform(kpt_type.begin(), kpt_type.end(), kpt_type.begin(), ::tolower);

    cv::Ptr<cv::Feature2D> backend;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    cv::Ptr<cv::AffineFeature> aff;

    if(kpt_type == "orb"){
      backend = cv::ORB::create();

      if(use_flann)
        matcher = cv::makePtr<cv::FlannBasedMatcher>(cv::makePtr<cv::flann::LshIndexParams>(6,12,1));
      else
        matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    }
    else if(kpt_type == "sift" || kpt_type == "sift-root"){
      backend = cv::SIFT::create();
      matcher = cv::DescriptorMatcher::create(use_flann ? 
                                              "FlannBased" :
                                              "BruteForce"
                                              );
    }
    else if(kpt_type == "brisk"){
      backend = cv::BRISK::create();
      if(use_flann)
        matcher = cv::makePtr<cv::FlannBasedMatcher>(cv::makePtr<cv::flann::LshIndexParams>(6,12,1));
      else
        matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    }
    else{
      throw std::runtime_error(kpt_type + " is not a known key-point backend");
    }
    aff = cv::AffineFeature::create(backend);

    std::cout <<  aff->getDefaultName() << "with backend " << kpt_type << std::endl;

    std::vector<cv::KeyPoint> kp0, kp1;
    cv::Mat desc0, desc1;

    // TODO: add timer
    auto t0 = std::chrono::high_resolution_clock::now();
    aff->detectAndCompute(im0, cv::Mat(), kp0, desc0);
    aff->detectAndCompute(im1, cv::Mat(), kp1, desc1);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto delta01 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    std::cout << "Number of keypoints im0: " << kp0.size() << std::endl;
    std::cout << "Number of keypoints im1: " << kp1.size() << std::endl;
    std::cout << "Time elapsed " << delta01.count()  << "ms" << std::endl;

    std::vector<std::vector<cv::DMatch>> raw_matches;
    std::vector<cv::Point2f> p0, p1;
    std::vector<float> distances;

    auto t2 = std::chrono::high_resolution_clock::now();
    matcher->knnMatch(desc0, desc1, raw_matches, 2);

    /* for(size_t i = 0; i < raw_matches.size(); ++i){ */
    for(const auto &cm: raw_matches){
      if(cm.size() == 2 && cm[0].distance < cm[1].distance * 0.75f){
        p0.push_back(kp0[cm[0].queryIdx].pt);
        p1.push_back(kp1[cm[1].queryIdx].pt);
        distances.push_back(cm[0].distance);
      }
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    auto delta23 = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2);
    std::cout << "Number of points after match: " << p0.size() << std::endl;
    std::cout << "Time elapsed " << delta23.count()  << "ms" << std::endl;

    std::vector<uchar> status;
    std::vector<std::tuple<cv::Point2f, cv::Point2f>> ppairs;

    auto t4 = std::chrono::high_resolution_clock::now();
    cv::Mat H = cv::findHomography(p0, p1, status, cv::RANSAC);
    int inliers = 0;

    for(size_t i = 0; i < status.size(); i++){
      if(status[i]){
        ppairs.push_back(std::make_tuple(p0[i], p1[i]));
        distances[inliers++] = distances[i];
      }
    }
    auto t5 = std::chrono::high_resolution_clock::now();
    auto delta45 = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2);
    std::cout << "Number of inliers: " << ppairs.size() << std::endl;
    std::cout << "Number of inliers/matched: " << ppairs.size() << "/" << p0.size() << std::endl;
    std::cout << "Time elapsed " << delta45.count()  << "ms" << std::endl;

    distances.resize(inliers);

  }
  catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}


#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <exiv2/exiv2.hpp>


#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <optional>
#include <tuple>
#include <typeinfo>
#include <thread>

namespace fs = std::filesystem;

static std::string exe_name(){
  std::string out;
  std::ifstream("/proc/self/comm") >> out;
  return out;
}

/* -- inputArgs struct */
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
          throw std::runtime_error("Unable to cast " + name_ + " to string");
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
          throw std::runtime_error("Unable to convert " + name_  + " <" + typeid(value).name() + "!=" + typeid(value_).name() + ">");  
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
            if(!parg->is_updated()){
              parg->set_value_from_string(current_arg_value);
            }
            else{
              help();
              throw std::runtime_error(current_arg_key + " is already set!");
            }
          }
        }
      }

      bool out = true;
      for(auto &parg: pargs)
        out &= parg->is_required() ? parg->is_updated(): true;

      return out;
    }
};

/* ---- input args end -- */ 

static std::tuple<int, int> get_image_size(const std::string &im_path){
  int imsizes[] = {-1, -1};
  if(fs::path(im_path).extension() == ".ppm"){
    std::ifstream ppmfile(im_path, std::ios::binary);

    std::string headerline;
    int read_headers = 0;
    while(read_headers < 2){
      std::getline(ppmfile, headerline);

      if(!headerline.empty() && headerline[0] == '#')
        continue;

      if(read_headers == 0){
        if(headerline != "P6"){
          throw std::runtime_error(im_path + "not of P6 type");
        }
      }
      else if(read_headers == 1){
        std::istringstream iss(headerline);
        for(int i = 0; i < 2; ++i){
          std::string size;
          std::getline(iss, size, ' ');
          imsizes[i] = std::stoi(size);
        }
      }
      read_headers++;
    }
    ppmfile.close();

  }
  else{
    auto tim = Exiv2::ImageFactory::open(im_path);
    tim->readMetadata();
    imsizes[0] = tim->pixelWidth();
    imsizes[1] = tim->pixelHeight();
  }

  return  std::make_tuple(imsizes[0], imsizes[1]);
}

static void write_model_matrix(const std::string &path,
                               const cv::Mat &model_matrix)
{
  assert(model_matrix.size().width == 3 && 
         model_matrix.size().height == 3);

  std::ofstream outfile(path);

  outfile << 
   model_matrix.at<float>(0,0) << " " <<
   model_matrix.at<float>(0,1) << " " <<
   model_matrix.at<float>(0,2) << "\n" <<

   model_matrix.at<float>(1,0) << " " <<
   model_matrix.at<float>(1,1) << " " <<
   model_matrix.at<float>(1,2) << "\n" <<

   model_matrix.at<float>(2,0) << " " <<
   model_matrix.at<float>(2,1) << " " <<
   model_matrix.at<float>(2,2) << "\n";

  outfile.close();
}

static void read_model_matrix(const std::string &path,
                              cv::Mat &model_matrix)
{
  assert(model_matrix.size().width == 3 && 
         model_matrix.size().height == 3);

  std::ifstream infile(path);

  infile >> model_matrix.at<float>(0,0) >> 
   model_matrix.at<float>(0,1) >> 
   model_matrix.at<float>(0,2) >> 

   model_matrix.at<float>(1,0) >> 
   model_matrix.at<float>(1,1) >> 
   model_matrix.at<float>(1,2) >> 

   model_matrix.at<float>(2,0) >> 
   model_matrix.at<float>(2,1) >> 
   model_matrix.at<float>(2,2); 

  infile.close();
}


void calcHomography(const std::vector<cv::Point2f> &p0, const std::vector<cv::Point2f> &p1, 
                      std::vector<std::tuple<cv::Point2f, cv::Point2f>> &inliers,
                      std::vector<float> &kptdistances,
                      std::vector<double> &model_distance,
                      cv::Mat &out
                      ){

    std::vector<uchar> status;
    out = cv::findHomography(p0, p1, status, cv::RANSAC);
    cv::Mat outInv = out.inv(); 

    int inliers_cnt = 0;
    for(size_t i = 0; i < status.size(); i++){
      if(status[i]){
        inliers.push_back(std::make_tuple(p0[i], p1[i]));
        kptdistances[inliers_cnt++] = kptdistances[i];

        cv::Mat p0_{(double)p0[i].x, (double)p0[i].y, 1.0};
        cv::Mat p1_{(double)p1[i].x, (double)p1[i].y, 1.0};

        cv::Mat phat01, phat10;
        cv::gemm(out, p0_, 1.0, cv::Mat(), 0.0, phat01);
        cv::gemm(outInv, p1_, 1.0, cv::Mat(), 0.0, phat10);

        phat01 = phat01 / phat01.at<double>(2);
        phat10 = phat10 / phat10.at<double>(2);

        cv::Mat diff01 = p1_ - phat01;
        cv::Mat diff10 = p0_ - phat10;

        cv::Mat d0{diff01.at<double>(0), diff01.at<double>(1)};
        cv::Mat d1{diff01.at<double>(0), diff01.at<double>(1)};

        model_distance.push_back((cv::norm(d0) +
                                  cv::norm(d1)) / 2);
      }
    }
}

cv::Mat calcFundamental(const std::vector<cv::Point2f> &p0, const std::vector<cv::Point2f> &p1, 
                      std::vector<std::tuple<cv::Point2f, cv::Point2f>> &inliers,
                      std::vector<float> &kptdistances,
                      std::vector<double> &model_distance,
                     cv::Mat &out 
                      ){

    std::vector<uchar> status;
    out = cv::findFundamentalMat(p0, p1, status, cv::RANSAC);

    int inliers_cnt = 0;
    for(size_t i = 0; i < status.size(); i++){
      if(status[i]){
        inliers.push_back(std::make_tuple(p0[i], p1[i]));
        kptdistances[inliers_cnt++] = kptdistances[i];

/*         cv::Mat p0_{(double)p0[i].x, (double)p0[i].y, 1.0}; */
/*         cv::Mat p1_{(double)p1[i].x, (double)p1[i].y, 1.0}; */

/*         cv::Mat phat01, phat10; */
/*         cv::gemm(out, p0_, 1.0, cv::Mat(), 0.0, phat01); */
/*         cv::gemm(outInv, p1_, 1.0, cv::Mat(), 0.0, phat10); */

/*         phat01 = phat01 / phat01.at<double>(0,2); */
/*         phat10 = phat10 / phat10.at<double>(0,2); */
/*         std::cout << phat01 << std::endl; */

/*         auto diff01 = p1_ - phat01; */
/*         auto diff10 = p0_ - phat10; */

        model_distance.push_back((cv::norm(diff01) + cv::norm(diff10)) / 2);
      }
    }
  return out;
}

int main(int argc, char *argv[])
{
  try {
    std::string im0_path, im1_path, mat_in,
      kpt_type = "orb",
      mat_out = "im0_im1_mat";

    bool use_flann = false;
    bool vis = true;

    appInputOpts opts;
    opts.add_argument("--image0", "-im0", "path to first image", &im0_path, true);
    opts.add_argument("--image1", "-im1", "path to second image", &im1_path, true);
    opts.add_argument("--keypoint", "-k", "kpt type", &kpt_type, false);
    opts.add_argument("--flann", "-f",    "if using flann or brute force matching", &use_flann, false);
    opts.add_argument("--vis", "-v",    "visualize the result of matching", &vis, false);
    opts.add_argument("--matrix_out", "-mo",    "(F/H)Matrix output path", &mat_out, false);
    opts.add_argument("--matrix_in", "-mi",    "(F/H)Matrix input path", &mat_in, false);

    if(!opts.parse_args(argc, argv)){
      opts.help();
      throw std::runtime_error("Non-valid input");
    }

    assert(fs::exists(fs::path(im0_path)));
    assert(fs::exists(fs::path(im1_path)));

    auto [w0, h0] = get_image_size(im0_path);
    auto [w1, h1] = get_image_size(im1_path);

    // display image
    cv::Mat disp = cv::Mat::zeros(cv::max(h0, h1), w0 + w1, CV_8U);
    cv::Mat im0 = cv::imread(im0_path, cv::IMREAD_GRAYSCALE);
    cv::Mat im1 = cv::imread(im1_path, cv::IMREAD_GRAYSCALE);

    im0.copyTo(cv::Mat(disp, cv::Rect(0, 0, w0, h0)));
    im1.copyTo(cv::Mat(disp, cv::Rect(w0, 0, w1, h1)));
    cv::cvtColor(disp, disp, cv::COLOR_GRAY2BGR);

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
    else if(kpt_type == "akaze"){
      backend = cv::AKAZE::create();
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

    if(kpt_type == "root-sift"){
      /* TODO: calculate root sift*/
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto delta01 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    std::cout << "Number of keypoints im0: " << kp0.size() << std::endl;
    std::cout << "Number of keypoints im1: " << kp1.size() << std::endl;
    std::cout << "Time elapsed " << delta01.count()  << "ms" << std::endl;

    // point matching
    std::vector<std::vector<cv::DMatch>> raw_matches;
    std::vector<cv::Point2f> p0, p1;
    std::vector<float> distances;
    auto t2 = std::chrono::high_resolution_clock::now();
    matcher->knnMatch(desc0, desc1, raw_matches, 2);
    /* for(size_t i = 0; i < raw_matches.size(); ++i){ */
    for(const auto &cm: raw_matches){
      if(cm.size() == 2 && cm[0].distance < cm[1].distance * 0.75){
        p0.push_back(kp0[cm[0].queryIdx].pt);
        p1.push_back(kp1[cm[0].trainIdx].pt);
        distances.push_back(cm[0].distance);
      }
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    auto delta23 = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2);
    std::cout << "Number of points after match: " << p0.size() << std::endl;
    std::cout << "Time elapsed " << delta23.count()  << "ms" << std::endl;

    // find homography and decide inliers
    /* std::vector<uchar> status; */
    std::vector<std::tuple<cv::Point2f, cv::Point2f>> ppairs;
    std::vector<double> model_error;


    auto t4 = std::chrono::high_resolution_clock::now();

    cv::Mat H;
    std::thread homThread(calcHomography, 
                          std::ref(p0), 
                          std::ref(p1), 
                          std::ref(ppairs), 
                          std::ref(distances),
                          std::ref(model_error),
                          std::ref(H)
                          );




    cv::Mat F;
    std::vector<double> model_error_fund;
    std::thread fundThread(calcFundamental, 
                          std::ref(p0), 
                          std::ref(p1), 
                          std::ref(ppairs), 
                          std::ref(model_error_fund),
                          std::ref(F)
                          );


    homThread.join();
    fundThread.join();

    double Hmodel_error = 0.0;
    std::for_each(model_error.begin(), model_error.end(), [&Hmodel_error](double me){
        Hmodel_error += me;
        });
    Hmodel_error /= model_error.size();

    /* double Hmodel_error = 0.0; */
    /* std::for_each(model_error.begin(), model_error.end(), [&Hmodel_error](double me){ */
    /*     Hmodel_error += me; */
    /*     }); */
    /* Hmodel_error /= model_error.size(); */

    auto t5 = std::chrono::high_resolution_clock::now();
    auto delta45 = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2);
    std::cout << "Number of inliers: " << ppairs.size() << std::endl;
    std::cout << "Number of inliers/matched: " << ppairs.size() << "/" << p0.size() << std::endl;
    std::cout << "Time elapsed " << delta45.count()  << "ms" << std::endl;
    std::cout << "Model error " << Hmodel_error << "pix" << std::endl;

    distances.resize(ppairs.size());

    // visualizing the mapping
    std::vector<cv::Point2f> corners(4);
    std::vector<cv::Point2f> corners_trans(4);
    corners[0] = cv::Point2f(0.f, 0.f);
    corners[1] = cv::Point2f((float)w0, 0.f);
    corners[2] = cv::Point2f((float)w0, (float)h0);
    corners[3] = cv::Point2f(0.f, (float)h0);

    std::vector<cv::Point2i> icorners;
    cv::perspectiveTransform(corners, corners_trans, H);
    cv::transform(corners_trans, corners_trans, cv::Matx23f(1, 0, (float)w0, 0, 1, 0));
    cv::Mat(corners_trans).convertTo(icorners, CV_32S);
    cv::polylines(disp, icorners, true, cv::Scalar(255, 255, 255));

    if(!mat_in.empty()){
      cv::Mat H_gt(3,3, CV_32F);
      std::vector<cv::Point2f> corners_gt(4);
      read_model_matrix(mat_in, H_gt);
      cv::perspectiveTransform(corners, corners_gt, H_gt);
      cv::transform(corners_gt, corners_gt, cv::Matx23f(1, 0, (float)w0, 0, 1, 0));
      cv::Mat(corners_gt).convertTo(icorners, CV_32S);
      cv::polylines(disp, icorners, true, cv::Scalar(0, 255, 0));
    }

    std::vector<int> indices(ppairs.size());
    cv::sortIdx(distances, indices, cv::SORT_EVERY_ROW + cv::SORT_ASCENDING);
    int thickness = 2;
    for(auto &index : indices){
      const cv::Point2f &pi1 = std::get<0>(ppairs[index]);
      const cv::Point2f &pi2 = std::get<1>(ppairs[index]);
      auto shifted_point = pi2 + cv::Point2f((float)w0, 0.f);
      cv::circle(disp, pi1, thickness, cv::Scalar(0, 255, 0), -1);
      cv::circle(disp, shifted_point, thickness, cv::Scalar(0, 255, 0), -1);
      cv::line(disp, pi1, shifted_point, cv::Scalar(0, 255, 0));
    }

    write_model_matrix(mat_out, H);

    cv::imshow("affine find_obj", disp);

    std::cout << "Press q to quit window\n";
    std::chrono::duration<double, std::milli> wait_time(20);

    while(cv::pollKey() != 'q'){
      std::this_thread::sleep_for(wait_time);
    }

  }
  catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
  }


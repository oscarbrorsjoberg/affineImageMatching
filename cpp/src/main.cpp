#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
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

#include "find_model.hpp" // find homography and fundamental
#include "genIO.hpp" // io args and write matrices

namespace fs = std::filesystem;

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
          /* size.erase(std::remove_if(size.begin(), size.end(), std::isspace), size.end()); */
          /* size.strio */
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

static void calc_root_norm(cv::Mat &desc){

  for(int i = 0; i < desc.rows; i++){
    const float *d0 = desc.ptr<float>(i);
    float l1norm = 0.0;

    // calculate l1 norm
    for(int n = 0; n < desc.cols; n++){
      l1norm += std::abs(d0[n]);
    }

    // normalize and sqrt
    float *d1 = desc.ptr<float>(i);
    for(int j = 0; j < desc.cols; j++){
      float val = d1[j];
      d1[j] = std::sqrt((val / (l1norm + 1e-8)));
    }
  }
}

void read_im_downsized(int max_size_width, int width,
                      cv::Mat &im, const std::string &im_path) {
    if(width / max_size_width > 8)
      im = cv::imread(im_path, cv::IMREAD_REDUCED_GRAYSCALE_8);
    else if(width / max_size_width > 4)
      im = cv::imread(im_path, cv::IMREAD_REDUCED_GRAYSCALE_4);
    else if(width / max_size_width > 2)
      im = cv::imread(im_path, cv::IMREAD_REDUCED_GRAYSCALE_2);
    else
      im = cv::imread(im_path, cv::IMREAD_GRAYSCALE);
}

int main(int argc, char *argv[])
{
  try {
    std::string im0_path, im1_path, mat_in,
      kpt_type = "orb",
      mat_out = "eufr_H",
      mat_out2 = "gt_H";

    bool use_flann = false;
    bool vis = true;


    appInputOpts opts("viewChange");

    opts.add_argument("--image0",     "-im0",  "path to first image", &im0_path, true);
    opts.add_argument("--image1",     "-im1",  "path to second image", &im1_path, true);
    opts.add_argument("--keypoint",   "-k",    "kpt type", &kpt_type, false);
    opts.add_argument("--flann",      "-f",    "if using flann or brute force matching", &use_flann, false);
    opts.add_argument("--vis",        "-v",    "visualize the result of matching", &vis, false);
    opts.add_argument("--matrix_out", "-mo",   "(F/H)Matrix output path", &mat_out, false);
    opts.add_argument("--matrix_in",  "-mi",   "(F/H)Matrix input path", &mat_in, false);
    opts.add_argument("--matrix_out2",  "-mo2",   "(F/H)Matrix output path 2", &mat_out2, false);

    if(!opts.parse_args(argc, argv)){
      opts.help();
      throw std::runtime_error("Non-valid input");
    }

    assert(fs::exists(fs::path(im0_path)));
    assert(fs::exists(fs::path(im1_path)));

    // pre read imag sizes so not to have to load the whole image to memory
    auto [w0, h0] = get_image_size(im0_path);
    auto [w1, h1] = get_image_size(im1_path);

    cv::Mat im0, im1;
    int max_im_width = 1060;

    read_im_downsized(max_im_width, w0, im0, im0_path);
    read_im_downsized(max_im_width, w1, im1, im1_path);

    std::cout << im0_path << " size " << im0.size().width << " "  << im0.size().height <<std::endl;
    std::cout << im1_path << " size " << im1.size().width << " " << im1.size().height <<std::endl;

    w0 = im0.size().width;
    h0 = im0.size().height;

    w1 = im1.size().width;
    h1 = im1.size().height;

    // display image
    cv::Mat disp_hom = cv::Mat::zeros(cv::max(h0, h1), w0 + w1, CV_8U);
    cv::Mat disp_epi = cv::Mat::zeros(cv::max(h0, h1), w0 + w1, CV_8U);

    im0.copyTo(cv::Mat(disp_hom, cv::Rect(0, 0, w0, h0)));
    im1.copyTo(cv::Mat(disp_hom, cv::Rect(w0, 0, w1, h1)));

    cv::cvtColor(disp_hom, disp_hom, cv::COLOR_GRAY2BGR);

    disp_hom.copyTo(disp_epi);

    std::transform(kpt_type.begin(), kpt_type.end(), kpt_type.begin(), ::tolower);

    cv::Ptr<cv::Feature2D> backend;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if(kpt_type == "orb"){
      backend = cv::ORB::create();

      if(use_flann)
        matcher = cv::makePtr<cv::FlannBasedMatcher>(cv::makePtr<cv::flann::LshIndexParams>(6,12,1));
      else
        matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    }
    else if(kpt_type == "sift" || kpt_type == "root-sift"){
      backend = cv::SIFT::create();
      matcher = cv::DescriptorMatcher::create(use_flann ? 
          "FlannBased" :
          "BruteForce"
          );
    }
    else if(kpt_type == "harris"){
      backend = cv::xfeatures2d::HarrisLaplaceFeatureDetector::create();
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
    auto aff = cv::AffineFeature::create(backend);
    /* auto aff = cv::xfeatures2d::AffineFeature2D::create(backend, cv::SIFT::create()); */

    std::cout << aff->getDefaultName() << "with backend " << kpt_type << std::endl;

    std::vector<cv::KeyPoint> kp0, kp1;
    cv::Mat desc0, desc1;

    // TODO: add timer
    auto t0 = std::chrono::high_resolution_clock::now();
    aff->detectAndCompute(im0, cv::Mat(), kp0, desc0);
    aff->detectAndCompute(im1, cv::Mat(), kp1, desc1);

    if(kpt_type == "root-sift"){
      calc_root_norm(desc0);
      calc_root_norm(desc1);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto delta01 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    std::cout << "Number of keypoints im0: " << kp0.size() << std::endl;
    std::cout << "Number of keypoints im1: " << kp1.size() << std::endl;
    std::cout << "Time elapsed " << delta01.count()  << "ms" << std::endl;

    // point matching
    std::vector<std::vector<cv::DMatch>> raw_matches;
    std::vector<cv::Point2d> p0, p1;
    std::vector<float> distances_hom;
    std::vector<float> distances_fund;
    auto t2 = std::chrono::high_resolution_clock::now();
    matcher->knnMatch(desc0, desc1, raw_matches, 2);
    for(const auto &cm: raw_matches){
      if(cm.size() == 2 && cm[0].distance < cm[1].distance * 0.75){
        p0.push_back(kp0[cm[0].queryIdx].pt);
        p1.push_back(kp1[cm[0].trainIdx].pt);
        distances_hom.push_back(cm[0].distance);
        distances_fund.push_back(cm[0].distance);
      }
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    auto delta23 = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2);
    std::cout << "Number of points after match: " << p0.size() << std::endl;
    std::cout << "Time elapsed " << delta23.count()  << "ms" << std::endl;


    // Normalization shouldn't be considered optional
    cv::Matx<double, 3, 3> hartleyNormp0, hartleyNormp1;

    std::vector<cv::Point2d> p0_norm = hartleyNorm(p0, hartleyNormp0);
    std::vector<cv::Point2d> p1_norm = hartleyNorm(p1, hartleyNormp1);


    auto t4 = std::chrono::high_resolution_clock::now();

    cv::Mat Hnorm;
    std::vector<double> model_error_hom;
    std::tuple<std::vector<cv::Point2d>, std::vector<cv::Point2d>> ppairs_hom = std::make_tuple(std::vector<cv::Point2d>{},
                                                                                                std::vector<cv::Point2d>{});
    std::thread homThread(calcHomography,
                          std::ref(p0_norm),
                          std::ref(p1_norm),
                          std::ref(p0),
                          std::ref(p1),
                          std::ref(ppairs_hom),
                          std::ref(distances_hom),
                          std::ref(model_error_hom),
                          std::ref(Hnorm)
                          );



    cv::Mat Fnorm;
    std::vector<double> model_error_fund;
    std::tuple<std::vector<cv::Point2d>, std::vector<cv::Point2d>> ppairs_fund = std::make_tuple(std::vector<cv::Point2d>{},
                                                                                                std::vector<cv::Point2d>{});
    std::thread fundThread(calcFundamental,
                          std::ref(p0_norm),
                          std::ref(p1_norm),
                          std::ref(p0),
                          std::ref(p1),
                          std::ref(ppairs_fund),
                          std::ref(distances_fund),
                          std::ref(model_error_fund),
                          std::ref(Fnorm)
                          );



    homThread.join();
    fundThread.join();

    cv::Mat H = hartleyNormp1.inv() * (Hnorm * hartleyNormp0);
    cv::Mat F = hartleyNormp1.t() * (Fnorm * hartleyNormp0);

    auto t5 = std::chrono::high_resolution_clock::now();

    double Hmodel_error = 0.0;
    double Hlargest_error = -1.0;
    std::for_each(model_error_hom.begin(), model_error_hom.end(), [&Hmodel_error, &Hlargest_error](double me){
        Hmodel_error += me;
        Hlargest_error = Hlargest_error > me ? Hlargest_error : me;
        });
    Hmodel_error /= model_error_hom.size();

    double Fmodel_error = 0.0;
    double Flargest_error = -1.0;
    std::for_each(model_error_fund.begin(), model_error_fund.end(), [&Fmodel_error, &Flargest_error](double me){
        Fmodel_error += me;
        Flargest_error = Flargest_error > me ? Flargest_error : me;
        });
    Fmodel_error /= model_error_fund.size();


    double Rh = (Hmodel_error / (Fmodel_error + Hmodel_error));
    std::cout << "Error Sh " << Hmodel_error << std::endl;
    std::cout << "Largest error Sh " << Hlargest_error << std::endl;
    std::cout << "Homography inliers " << std::get<0>(ppairs_hom).size() << std::endl;

    std::cout << "Error Sf " << Fmodel_error << std::endl;
    std::cout << "Largest error Sf " << Flargest_error << std::endl;
    std::cout << "Fundamental inliers " << std::get<0>(ppairs_fund).size() << std::endl;

    /* std::tuple<std::vector<cv::Point2d>, std::vector<cv::Point2d>> ppairs_fund2 = std::make_tuple(std::vector<cv::Point2d>{}, */
    /*                                                                                             std::vector<cv::Point2d>{}); */

    /* cv::correctMatches(F, std::get<0>(ppairs_fund), std::get<1>(ppairs_fund), std::get<0>(ppairs_fund2), std::get<1>(ppairs_fund2)); */

    /* std::cout << "Fundamental inliers " << std::get<0>(ppairs_fund2).size() << std::endl; */




    std::cout << "Error Rh " << Rh << std::endl;
    auto &ppairs = Rh  > 0.45 ? ppairs_hom : ppairs_fund; 
    bool homSelected = Rh  > 0.45;

    if(homSelected)
      std::cout << "hom selected \n";
    else
      std::cout << "fund selected \n";

    auto delta45 = std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4);
    std::cout << "Number of inliers: " << std::get<0>(ppairs).size() << std::endl;
    std::cout << "Number of inliers/matched: " << std::get<0>(ppairs).size() << "/" 
      << p0.size() << std::endl;
    std::cout << "Time elapsed " << delta45.count()  << "ms" << std::endl;
    std::cout << "Model error " << Hmodel_error << "pix" << std::endl;

    distances_hom.resize(std::get<0>(ppairs_hom).size());
    distances_fund.resize(std::get<0>(ppairs_fund).size());

    // visualizing the homography mapping
    std::vector<cv::Point2f> corners(4);
    std::vector<cv::Point2f> corners_trans(4);
    corners[0] = cv::Point2f(0.f, 0.f); 
    corners[1] = cv::Point2f((float)w0, 0.f); corners[2] = cv::Point2f((float)w0, (float)h0);
    corners[3] = cv::Point2f(0.f, (float)h0);

    std::vector<cv::Point2i> icorners;
    cv::perspectiveTransform(corners, corners_trans, H);
    cv::transform(corners_trans, corners_trans, cv::Matx23f(1, 0, (float)w0, 0, 1, 0));
    cv::Mat(corners_trans).convertTo(icorners, CV_32S);
    cv::polylines(disp_hom, icorners, true, cv::Scalar(255, 255, 255));

    if(!mat_in.empty()){
      cv::Mat H_gt(3,3, CV_64F);
      std::vector<cv::Point2f> corners_gt(4);
      read_model_matrix(mat_in, H_gt);
      cv::perspectiveTransform(corners, corners_gt, H_gt);
      cv::transform(corners_gt, corners_gt, cv::Matx23f(1, 0, (float)w0, 0, 1, 0));
      cv::Mat(corners_gt).convertTo(icorners, CV_32S);
      cv::polylines(disp_hom, icorners, true, cv::Scalar(0, 255, 0));
      std::cout << "writing matrix 2 \n";
      write_model_matrix(mat_out2, H_gt);

    }

    // homography
    std::vector<int> indices(std::get<0>(ppairs_hom).size());
    cv::sortIdx(distances_hom, indices, cv::SORT_EVERY_ROW + cv::SORT_ASCENDING);
    int thickness = 2;
    for(auto &index : indices){
      const cv::Point2d &pi0 = std::get<0>(ppairs_hom)[index];
      const cv::Point2d &pi1 = std::get<1>(ppairs_hom)[index];
      auto shifted_point = pi1 + cv::Point2d((double)w0, 0.f);
      cv::circle(disp_hom, pi0, thickness, cv::Scalar(0, 255, 0), -1);
      cv::circle(disp_hom, shifted_point, thickness, cv::Scalar(0, 255, 0), -1);
      cv::line(disp_hom, pi0, shifted_point, cv::Scalar(0, 255, 0));
    }

    // points for F and epipolar lines

    std::vector<int> indices_fund(std::get<0>(ppairs_fund).size());
    cv::sortIdx(distances_fund, indices_fund, cv::SORT_EVERY_ROW + cv::SORT_ASCENDING);
    for(auto &index : indices){
      const cv::Point2d &pi0 = std::get<0>(ppairs_fund)[index];
      const cv::Point2d &pi1 = std::get<1>(ppairs_fund)[index];

      /* const cv::Point2d &pi0_corr = std::get<0>(ppairs_fund2)[index]; */
      /* const cv::Point2d &pi1_corr = std::get<1>(ppairs_fund2)[index]; */

      auto shifted_point = pi1 + cv::Point2d((double)w0, 0.f);
      /* auto shifted_point2 = pi1_corr + cv::Point2d((double)w0, 0.f); */

      cv::circle(disp_epi, pi0, thickness, cv::Scalar(255, 0, 0), -1);
      cv::circle(disp_epi, shifted_point, thickness, cv::Scalar(255, 0, 0), -1);

      /* cv::circle(disp_epi, pi0_corr, thickness, cv::Scalar(255, 0, 255), -1); */
      /* cv::circle(disp_epi, shifted_point2, thickness, cv::Scalar(255, 0, 255), -1); */

    }

    std::vector<cv::Point3d> lines0(std::get<1>(ppairs_fund).size());
    std::vector<cv::Point3d> lines1(std::get<0>(ppairs_fund).size());
    
    cv::computeCorrespondEpilines(std::get<1>(ppairs_fund), 2, F, lines0);
    cv::computeCorrespondEpilines(std::get<0>(ppairs_fund), 1, F, lines1);

    for(auto &line: lines0){
      std::vector<cv::Point2d> eCorners(2);
      std::vector<cv::Point2i> ieCorners;

      eCorners[0] = {0, -line.z / line.y };
      eCorners[1] = {(float)im0.size().width, 
                     (((-line.x * (float)im0.size().width)) - line.z ) / line.y};


      cv::Mat(eCorners).convertTo(ieCorners, CV_32S);

      cv::polylines(disp_epi, ieCorners, true, cv::Scalar(0, 255, 0));
    }

    for(auto &line: lines1){
      std::vector<cv::Point2d> eCorners(2);
      std::vector<cv::Point2i> ieCorners;

      eCorners[0] = {0, -line.z / line.y };
      eCorners[1] = {(float)im1.size().width, 
                     (((-line.x * (float)im1.size().width)) - line.z ) / line.y};


      cv::transform(eCorners, eCorners, cv::Matx23f(1, 0, (float)w0, 0, 1, 0));
      cv::Mat(eCorners).convertTo(ieCorners, CV_32S);

      cv::polylines(disp_epi, ieCorners, true, cv::Scalar(0, 255, 0));
    }

    write_model_matrix(mat_out, H);

    if(vis){
      cv::imshow("epipolar lines", disp_epi);

      std::cout << "Press q to quit window\n";
      std::cout << "Press h to vis hom match\n";
      std::cout << "Press e to vis epipolar lines\n";

      std::chrono::duration<double, std::milli> wait_time(20);

      int k = ' ';
      while(k != 'q'){
        if(k == 'h')
          cv::imshow("hom match", disp_hom);
        else if(k == 'e')
          cv::imshow("epipolar lines", disp_epi);
        /* else if(k == 'r') */
        /*   cv::imshow("rectified result", disp_rect); */

        std::this_thread::sleep_for(wait_time);
        k = cv::pollKey();
      }
    }


  }
  catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
  }


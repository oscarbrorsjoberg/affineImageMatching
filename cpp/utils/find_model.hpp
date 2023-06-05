#ifndef find_model_hpp
#define find_model_hpp

#include <opencv2/core.hpp>

void calcHomography(const std::vector<cv::Point2d> &p0n, 
    const std::vector<cv::Point2d> &p1n, 
    const std::vector<cv::Point2d> &p0, 
    const std::vector<cv::Point2d> &p1,
    std::tuple<std::vector<cv::Point2d>, std::vector<cv::Point2d>> &inliers,
    std::vector<float> &kptdistances,
    std::vector<double> &model_distance,
    cv::Mat &out
    );

void calcFundamental(const std::vector<cv::Point2d> &p0n, const std::vector<cv::Point2d> &p1n, 
    const std::vector<cv::Point2d> &p0, const std::vector<cv::Point2d> &p1,
    std::tuple<std::vector<cv::Point2d>, std::vector<cv::Point2d>> &inliers,
    std::vector<float> &kptdistances,
    std::vector<double> &model_distance,
    cv::Mat &out 
    );

std::vector<cv::Point2d> 
hartleyNorm(const std::vector<cv::Point2d> &input, cv::Matx<double, 3, 3> &out_trans);

#endif // !find_model_hpp

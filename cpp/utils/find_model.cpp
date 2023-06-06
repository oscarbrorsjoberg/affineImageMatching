

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

#include "find_model.hpp"

constexpr double ThreshOutlHom = 5.99; // from orb-slam
constexpr double ThreshOutlFund = 3.84; // from orb-slam
/**
 * Calc homography
 *
 * */
void calcHomography(const std::vector<cv::Point2d> &p0n, 
                    const std::vector<cv::Point2d> &p1n, 
                    const std::vector<cv::Point2d> &p0, 
                    const std::vector<cv::Point2d> &p1,
                    std::tuple<std::vector<cv::Point2d>, std::vector<cv::Point2d>> &inliers,
                    std::vector<float> &kptdistances,
                    std::vector<double> &model_distance,
                    cv::Mat &out
                    ){

    std::vector<uchar> status;
    out = cv::findHomography(p0n, p1n, status, cv::RANSAC, ThreshOutlHom);
    cv::Mat outInv = out.inv(); 

    int inliers_cnt = 0;
    for(size_t i = 0; i < status.size(); i++){
      if(status[i]){

        std::get<0>(inliers).push_back(p0[i]); 
        std::get<1>(inliers).push_back(p1[i]); 

        kptdistances[inliers_cnt++] = kptdistances[i];

        // to homogenous
        cv::Mat p0_{(double)p0n[i].x, (double)p0n[i].y, 1.0};
        cv::Mat p1_{(double)p1n[i].x, (double)p1n[i].y, 1.0};

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

/**
 * Calc fundamental
 *
 * */

void calcFundamental(const std::vector<cv::Point2d> &p0n, const std::vector<cv::Point2d> &p1n, 
                     const std::vector<cv::Point2d> &p0, const std::vector<cv::Point2d> &p1,
                     std::tuple<std::vector<cv::Point2d>, std::vector<cv::Point2d>> &inliers,
                     std::vector<float> &kptdistances,
                     std::vector<double> &model_distance,
                     cv::Mat &out 
                     ){

    std::vector<uchar> status;
    out = cv::findFundamentalMat(p0n, p1n, status, cv::FM_RANSAC + cv::FM_8POINT, ThreshOutlFund);

    int inliers_cnt = 0;
    for(size_t i = 0; i < status.size(); i++){
      if(status[i]){

        std::get<0>(inliers).push_back(p0[i]); 
        std::get<1>(inliers).push_back(p1[i]); 

        kptdistances[inliers_cnt++] = kptdistances[i];
        cv::Mat p0_{(double)p0n[i].x, (double)p0n[i].y, 1.0};
        cv::Mat p1_{(double)p1n[i].x, (double)p1n[i].y, 1.0};

        model_distance.push_back(cv::sampsonDistance(p0_, p1_, out));
      }
    }
}

/**
 * Hartley Normalization
 *
 * */

std::vector<cv::Point2d> hartleyNorm(const std::vector<cv::Point2d> &input, cv::Matx<double, 3, 3> &out_trans)
{
  std::vector<cv::Point3d> homo(input.size());
  std::vector<cv::Point2d> out(input.size());


  double centroid_x = 0.0;
  double centroid_y = 0.0;
  int idx = 0;
  std::for_each(input.begin(), input.end(), [&centroid_x, &centroid_y, &homo, &idx](auto &pnt){
      centroid_x += pnt.x;
      centroid_y += pnt.y;
      homo[idx++] = {pnt.x, pnt.y, 1.0};
    });

  centroid_x /= input.size();
  centroid_y /= input.size();
  out_trans = cv::Matx33d(1.0 / std::sqrt(2), 0.0,  -centroid_x,
                          0.0, 1.0 / std::sqrt(2) , -centroid_y,
                          0.0, 0.0, 1.0
                                      );

  cv::transform(homo, homo, out_trans);

  cv::convertPointsFromHomogeneous(homo, out);

  return out;
}

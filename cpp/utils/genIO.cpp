#include <opencv2/core.hpp>

#include <iostream>
#include <fstream>

#include "genIO.hpp"



void write_model_matrix(const std::string &path,
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

void read_model_matrix(const std::string &path,
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

// appInputOpts ---------------
void appInputOpts::help(){
      std::cout << " This is " << app_name << " a simple copy of OpenCVs asift. " << std::endl;
      std::cout << " Input parameters:" << std::endl;
      for(auto &parg: pargs){
        std::cout << "\t" << parg->name_ << " " << parg->short_name_ << " | " << parg->description_  << " | " ;
        std::cout << parg->value_str() << "\n";
      }
    }


bool appInputOpts::parse_args(int argc, char *args[]){
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

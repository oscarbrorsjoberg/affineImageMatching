#ifndef genIO_hpp
#define genIO_hpp

#include <opencv2/core.hpp>
#include <fstream>
#include <istream>

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
          throw std::runtime_error("Unable to convert " + name_  
              + " <" + typeid(value).name() + "!=" + typeid(value_).name() + ">");  
      }
    }
};

class appInputOpts {

  private:
    std::vector<std::unique_ptr<inputArgBase>> pargs;
    std::string app_name;

  public:

    appInputOpts(const std::string &appname):
      app_name(appname)
  {}

    template<typename T>
    void add_argument(const std::string &long_name, const std::string &short_name,
          const std::string &description, T *var, bool required)
  {
    pargs.push_back(std::make_unique<inputArg<T>>( 
          long_name, short_name, description, 
          var, required));
  }

    void help();
    bool parse_args(int argc, char *args[]);
};

void write_model_matrix(const std::string &path,
                               const cv::Mat &model_matrix);

void read_model_matrix(const std::string &path,
                              cv::Mat &model_matrix);
#endif // !genIO_hpp

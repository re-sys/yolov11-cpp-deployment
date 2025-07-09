#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include "tools/Config.hpp"

struct Segmentation {
    int id = 0;
    float confidence = 0.0;
    cv::Rect bbox;
    cv::Mat mask;
};

struct ImageInfo {
    cv::Size raw_size;
    cv::Vec4d trans;
};

class YOLOSeg {
public:
    YOLOSeg();
    ~YOLOSeg();
    
    bool initialize(const std::string& model_path, const std::string& classes_path, bool use_gpu = false);
    std::vector<Segmentation> detect(const cv::Mat& image, float conf_threshold = DEFAULT_CONF_THRESHOLD, float mask_threshold = 0.5f);
    void draw_result(cv::Mat& image, const std::vector<Segmentation>& results);
    bool process_video(const std::string& video_path, const std::string& output_path = "");
    bool process_image(const std::string& image_path, const std::string& output_path = "");
    
private:
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    
    std::vector<std::string> class_names_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    
    int seg_channels_;
    int seg_width_;
    int seg_height_;
    int net_width_;
    int net_height_;
    
    bool load_class_names(const std::string& classes_path);
    ImageInfo calculate_transform(const cv::Mat& image);
    void get_mask(const cv::Mat& mask_info, const cv::Mat& mask_data, 
                  const ImageInfo& para, cv::Rect bound, cv::Mat& mask_out,
                  float mask_threshold);
    void decode_output(cv::Mat& output0, cv::Mat& output1, 
                      const ImageInfo& para, std::vector<Segmentation>& output,
                      float conf_threshold, float mask_threshold);
    cv::Scalar get_random_color(int class_id);
    cv::Mat letterbox(const cv::Mat& image, int target_width, int target_height, cv::Vec4d& trans, cv::Scalar color = cv::Scalar(114, 114, 114));
}; 
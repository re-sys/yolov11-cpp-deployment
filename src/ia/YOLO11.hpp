#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>
#include "tools/Config.hpp"

struct Detection {
    cv::Rect bbox;
    float confidence;
    int class_id;
    std::string class_name;
};

class YOLO11 {
public:
    YOLO11();
    ~YOLO11();
    
    bool initialize(const std::string& model_path, const std::string& classes_path, bool use_gpu = false);
    std::vector<Detection> detect(const cv::Mat& image, float conf_threshold = DEFAULT_CONF_THRESHOLD, float iou_threshold = DEFAULT_IOU_THRESHOLD);
    void draw_detections(cv::Mat& image, const std::vector<Detection>& detections);
    
private:
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    
    std::vector<std::string> class_names_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    
    int input_width_;
    int input_height_;
    
    bool load_class_names(const std::string& classes_path);
    cv::Mat preprocess(const cv::Mat& image);
    std::vector<Detection> postprocess(const std::vector<float>& output, 
                                      int original_width, int original_height,
                                      float conf_threshold, float iou_threshold);
    std::vector<int> non_max_suppression(const std::vector<cv::Rect>& boxes, 
                                        const std::vector<float>& scores, 
                                        float iou_threshold);
    float calculate_iou(const cv::Rect& box1, const cv::Rect& box2);
}; 
#include "YOLO11.hpp"
#include <fstream>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <cstring>

YOLO11::YOLO11() : input_width_(DEFAULT_INPUT_WIDTH), input_height_(DEFAULT_INPUT_HEIGHT) {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "YOLO11");
}

YOLO11::~YOLO11() {
    // Free allocated input/output names
    for (auto name : input_names_) {
        free(const_cast<char*>(name));
    }
    for (auto name : output_names_) {
        free(const_cast<char*>(name));
    }
}

bool YOLO11::initialize(const std::string& model_path, const std::string& classes_path, bool use_gpu) {
    try {
        // Load class names
        if (!load_class_names(classes_path)) {
            std::cerr << "Failed to load class names from: " << classes_path << std::endl;
            return false;
        }
        
        // Create session options
        session_options_ = std::make_unique<Ort::SessionOptions>();
        session_options_->SetIntraOpNumThreads(1);
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        if (use_gpu) {
            try {
                OrtCUDAProviderOptions cuda_options{};
                session_options_->AppendExecutionProvider_CUDA(cuda_options);
                DEBUG_PRINT("GPU acceleration enabled");
            } catch (const std::exception& e) {
                std::cerr << "Failed to enable GPU acceleration: " << e.what() << std::endl;
                std::cerr << "Falling back to CPU" << std::endl;
            }
        }
        
        // Create session
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), *session_options_);
        
        // Get input and output names
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Get input and output names using static storage
        size_t num_input_nodes = session_->GetInputCount();
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = session_->GetInputNameAllocated(i, allocator);
            input_names_.push_back(strdup(input_name.get()));
            
            // Get input shape
            auto input_shape = session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            if (input_shape.size() >= 4) {
                input_height_ = static_cast<int>(input_shape[2]);
                input_width_ = static_cast<int>(input_shape[3]);
            }
        }
        
        // Output info
        size_t num_output_nodes = session_->GetOutputCount();
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = session_->GetOutputNameAllocated(i, allocator);
            output_names_.push_back(strdup(output_name.get()));
        }
        
        DEBUG_PRINT("Model loaded successfully");
        DEBUG_PRINT("Input size: " << input_width_ << "x" << input_height_);
        DEBUG_PRINT("Number of classes: " << class_names_.size());
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize YOLO11: " << e.what() << std::endl;
        return false;
    }
}

bool YOLO11::load_class_names(const std::string& classes_path) {
    std::ifstream file(classes_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open classes file: " << classes_path << std::endl;
        return false;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            class_names_.push_back(line);
        }
    }
    
    return !class_names_.empty();
}

cv::Mat YOLO11::preprocess(const cv::Mat& image) {
    cv::Mat resized, normalized;
    
    // Resize image
    cv::resize(image, resized, cv::Size(input_width_, input_height_));
    
    // Convert BGR to RGB
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    
    // Normalize to [0, 1]
    resized.convertTo(normalized, CV_32F, 1.0 / 255.0);
    
    return normalized;
}

std::vector<Detection> YOLO11::detect(const cv::Mat& image, float conf_threshold, float iou_threshold) {
    if (!session_) {
        std::cerr << "Model not initialized" << std::endl;
        return {};
    }
    
    START_TIMER(total_detection);
    
    int original_width = image.cols;
    int original_height = image.rows;
    
    // Preprocess
    START_TIMER(preprocessing);
    cv::Mat processed = preprocess(image);
    END_TIMER(preprocessing);
    
    // Prepare input tensor
    std::vector<float> input_data;
    input_data.reserve(input_width_ * input_height_ * 3);
    
    // Convert HWC to CHW format
    std::vector<cv::Mat> channels;
    cv::split(processed, channels);
    
    for (const auto& channel : channels) {
        float* data = reinterpret_cast<float*>(channel.data);
        input_data.insert(input_data.end(), data, data + input_width_ * input_height_);
    }
    
    // Create input tensor
    std::vector<int64_t> input_shape = {1, 3, input_height_, input_width_};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());
    
    // Run inference
    START_TIMER(inference);
    auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, 
                                       input_names_.data(), &input_tensor, 1,
                                       output_names_.data(), output_names_.size());
    END_TIMER(inference);
    
    // Get output data
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    
    size_t output_size = 1;
    for (auto dim : output_shape) {
        output_size *= dim;
    }
    
    std::vector<float> output(output_data, output_data + output_size);
    
    // Postprocess
    START_TIMER(postprocessing);
    auto detections = postprocess(output, original_width, original_height, conf_threshold, iou_threshold);
    END_TIMER(postprocessing);
    
    END_TIMER(total_detection);
    
    DEBUG_PRINT("Detected " << detections.size() << " objects");
    
    return detections;
}

std::vector<Detection> YOLO11::postprocess(const std::vector<float>& output, 
                                          int original_width, int original_height,
                                          float conf_threshold, float iou_threshold) {
    std::vector<Detection> detections;
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    
    // Based on model info: output shape [1, 5, 8400]
    // This means: 8400 detections, each with [x, y, w, h, confidence]
    int num_detections = 8400;
    int elements_per_detection = 5; // x, y, w, h, confidence
    
    float x_scale = static_cast<float>(original_width) / input_width_;
    float y_scale = static_cast<float>(original_height) / input_height_;
    
    for (int i = 0; i < num_detections; ++i) {
        // For output shape [1, 5, 8400], data is organized as:
        // output[0*8400 + i] = x_center
        // output[1*8400 + i] = y_center  
        // output[2*8400 + i] = width
        // output[3*8400 + i] = height
        // output[4*8400 + i] = confidence
        
        float center_x = output[0 * num_detections + i];
        float center_y = output[1 * num_detections + i];
        float width = output[2 * num_detections + i];
        float height = output[3 * num_detections + i];
        float confidence = output[4 * num_detections + i];
        
        if (confidence >= conf_threshold) {
            // Convert center format to corner format and scale back to original image
            int x = static_cast<int>((center_x - width / 2.0f) * x_scale);
            int y = static_cast<int>((center_y - height / 2.0f) * y_scale);
            int w = static_cast<int>(width * x_scale);
            int h = static_cast<int>(height * y_scale);
            
            // Clamp coordinates
            x = std::max(0, std::min(x, original_width - 1));
            y = std::max(0, std::min(y, original_height - 1));
            w = std::max(1, std::min(w, original_width - x));
            h = std::max(1, std::min(h, original_height - y));
            
            boxes.emplace_back(x, y, w, h);
            confidences.push_back(confidence);
            class_ids.push_back(0); // Only one class
        }
    }
    
    // Apply Non-Maximum Suppression
    std::vector<int> indices = non_max_suppression(boxes, confidences, iou_threshold);
    
    for (int idx : indices) {
        Detection det;
        det.bbox = boxes[idx];
        det.confidence = confidences[idx];
        det.class_id = class_ids[idx];
        det.class_name = (class_ids[idx] < class_names_.size()) ? class_names_[class_ids[idx]] : "Unknown";
        detections.push_back(det);
    }
    
    return detections;
}

std::vector<int> YOLO11::non_max_suppression(const std::vector<cv::Rect>& boxes, 
                                            const std::vector<float>& scores, 
                                            float iou_threshold) {
    std::vector<int> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    // Sort by confidence (descending)
    std::sort(indices.begin(), indices.end(), [&scores](int i1, int i2) {
        return scores[i1] > scores[i2];
    });
    
    std::vector<int> result;
    std::vector<bool> suppressed(indices.size(), false);
    
    for (size_t i = 0; i < indices.size(); ++i) {
        if (suppressed[i]) continue;
        
        int idx = indices[i];
        result.push_back(idx);
        
        for (size_t j = i + 1; j < indices.size(); ++j) {
            if (suppressed[j]) continue;
            
            int idx2 = indices[j];
            float iou = calculate_iou(boxes[idx], boxes[idx2]);
            
            if (iou > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }
    
    return result;
}

float YOLO11::calculate_iou(const cv::Rect& box1, const cv::Rect& box2) {
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);
    
    if (x2 <= x1 || y2 <= y1) return 0.0f;
    
    float intersection = (x2 - x1) * (y2 - y1);
    float area1 = box1.width * box1.height;
    float area2 = box2.width * box2.height;
    float union_area = area1 + area2 - intersection;
    
    return intersection / union_area;
}

void YOLO11::draw_detections(cv::Mat& image, const std::vector<Detection>& detections) {
    for (const auto& det : detections) {
        // Draw semi-transparent filled rectangle
        cv::Mat overlay = image.clone();
        cv::rectangle(overlay, det.bbox, cv::Scalar(BOX_COLOR_B, BOX_COLOR_G, BOX_COLOR_R), -1);
        cv::addWeighted(image, 1.0 - BOX_ALPHA, overlay, BOX_ALPHA, 0, image);
        
        // Draw border
        cv::rectangle(image, det.bbox, cv::Scalar(BOX_COLOR_B, BOX_COLOR_G, BOX_COLOR_R), 2);
        
        // Draw label
        std::string label = det.class_name + " " + std::to_string(static_cast<int>(det.confidence * 100)) + "%";
        
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        
        cv::Point label_pos(det.bbox.x, det.bbox.y - 5);
        if (label_pos.y < text_size.height) {
            label_pos.y = det.bbox.y + text_size.height + 5;
        }
        
        // Draw label background
        cv::rectangle(image, 
                     cv::Point(label_pos.x, label_pos.y - text_size.height),
                     cv::Point(label_pos.x + text_size.width, label_pos.y + baseline),
                     cv::Scalar(BOX_COLOR_B, BOX_COLOR_G, BOX_COLOR_R), -1);
        
        // Draw label text
        cv::putText(image, label, label_pos, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
} 
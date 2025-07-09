#include "YOLOSeg.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>

YOLOSeg::YOLOSeg() 
    : seg_channels_(32), seg_width_(160), seg_height_(160), 
      net_width_(640), net_height_(640) {
}

YOLOSeg::~YOLOSeg() {
}

bool YOLOSeg::initialize(const std::string& model_path, const std::string& classes_path, bool use_gpu) {
    // Load class names
    if (!load_class_names(classes_path)) {
        std::cerr << "Failed to load class names from: " << classes_path << std::endl;
        return false;
    }
    
    // Initialize ONNX Runtime
    env_ = std::make_unique<Ort::Env>(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "yolov8_seg");
    session_options_ = std::make_unique<Ort::SessionOptions>();
    session_options_->SetIntraOpNumThreads(1);
    session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    if (use_gpu) {
        try {
            OrtCUDAProviderOptions cuda_options{};
            session_options_->AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "GPU acceleration enabled for YOLOSeg" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to enable GPU acceleration: " << e.what() << std::endl;
            std::cerr << "Falling back to CPU" << std::endl;
        }
    }
    
    // Create session
    try {
        session_ = std::make_unique<Ort::Session>(*env_, 
            model_path.c_str(), 
            *session_options_);
    } catch (const Ort::Exception& e) {
        std::cerr << "Failed to create ONNX session: " << e.what() << std::endl;
        return false;
    }
    
    // Set input/output names
    input_names_.push_back("images");
    output_names_.push_back("output0");
    output_names_.push_back("output1");
    
    std::cout << "YOLOSeg initialized successfully with " << class_names_.size() << " classes" << std::endl;
    return true;
}

bool YOLOSeg::load_class_names(const std::string& classes_path) {
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

ImageInfo YOLOSeg::calculate_transform(const cv::Mat& image) {
    ImageInfo info;
    info.raw_size = image.size();
    info.trans = cv::Vec4d(640.0 / image.cols, 640.0 / image.rows, 0, 0);
    return info;
}

void YOLOSeg::get_mask(const cv::Mat& mask_info, const cv::Mat& mask_data, 
                      const ImageInfo& para, cv::Rect bound, cv::Mat& mask_out,
                      float mask_threshold) {
    cv::Vec4f trans = para.trans;
    int r_x = floor((bound.x * trans[0] + trans[2]) / net_width_ * seg_width_);
    int r_y = floor((bound.y * trans[1] + trans[3]) / net_height_ * seg_height_);
    int r_w = ceil(((bound.x + bound.width) * trans[0] + trans[2]) / net_width_ * seg_width_) - r_x;
    int r_h = ceil(((bound.y + bound.height) * trans[1] + trans[3]) / net_height_ * seg_height_) - r_y;
    
    r_w = MAX(r_w, 1);
    r_h = MAX(r_h, 1);
    
    if (r_x + r_w > seg_width_) {
        seg_width_ - r_x > 0 ? r_w = seg_width_ - r_x : r_x -= 1;
    }
    if (r_y + r_h > seg_height_) {
        seg_height_ - r_y > 0 ? r_h = seg_height_ - r_y : r_y -= 1;
    }
    
    std::vector<cv::Range> roi_ranges = { 
        cv::Range(0, 1), 
        cv::Range::all(), 
        cv::Range(r_y, r_h + r_y), 
        cv::Range(r_x, r_w + r_x) 
    };
    
    cv::Mat temp_mask = mask_data(roi_ranges).clone();
    cv::Mat protos = temp_mask.reshape(0, {seg_channels_, r_w * r_h});
    cv::Mat matmul_res = (mask_info * protos).t();
    cv::Mat masks_feature = matmul_res.reshape(1, {r_h, r_w});
    cv::Mat dest;
    cv::exp(-masks_feature, dest); // sigmoid
    dest = 1.0 / (1.0 + dest);
    
    int left = floor((net_width_ / seg_width_ * r_x - trans[2]) / trans[0]);
    int top = floor((net_height_ / seg_height_ * r_y - trans[3]) / trans[1]);
    int width = ceil(net_width_ / seg_width_ * r_w / trans[0]);
    int height = ceil(net_height_ / seg_height_ * r_h / trans[1]);
    
    cv::Mat mask;
    cv::resize(dest, mask, cv::Size(width, height));
    mask_out = mask(bound - cv::Point(left, top)) > mask_threshold;
}

void YOLOSeg::decode_output(cv::Mat& output0, cv::Mat& output1, 
                           const ImageInfo& para, std::vector<Segmentation>& output,
                           float conf_threshold, float mask_threshold) {
    output.clear();
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<float>> masks;
    
    int data_width = class_names_.size() + 4 + 32;
    int rows = output0.rows;
    float* pdata = (float*)output0.data;
    
    for (int r = 0; r < rows; ++r) {
        cv::Mat scores(1, class_names_.size(), CV_32FC1, pdata + 4);
        cv::Point class_id;
        double max_score;
        cv::minMaxLoc(scores, 0, &max_score, 0, &class_id);
        
        if (max_score >= conf_threshold) {
            masks.push_back(std::vector<float>(pdata + 4 + class_names_.size(), pdata + data_width));
            float w = pdata[2] / para.trans[0];
            float h = pdata[3] / para.trans[1];
            int left = MAX(int((pdata[0] - para.trans[2]) / para.trans[0] - 0.5 * w + 0.5), 0);
            int top = MAX(int((pdata[1] - para.trans[3]) / para.trans[1] - 0.5 * h + 0.5), 0);
            
            class_ids.push_back(class_id.x);
            confidences.push_back(max_score);
            boxes.push_back(cv::Rect(left, top, int(w + 0.5), int(h + 0.5)));
        }
        pdata += data_width; // next line
    }
    
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, mask_threshold, nms_result);
    
    for (int i = 0; i < nms_result.size(); ++i) {
        int idx = nms_result[i];
        boxes[idx] = boxes[idx] & cv::Rect(0, 0, para.raw_size.width, para.raw_size.height);
        
        Segmentation result;
        result.id = class_ids[idx];
        result.confidence = confidences[idx];
        result.bbox = boxes[idx];
        
        get_mask(cv::Mat(masks[idx]).t(), output1, para, boxes[idx], result.mask, mask_threshold);
        output.push_back(result);
    }
}

cv::Mat YOLOSeg::letterbox(const cv::Mat& image, int target_width, int target_height, cv::Vec4d& trans, cv::Scalar color) {
    int img_w = image.cols, img_h = image.rows;
    float r = std::min(target_width / (img_w*1.0f), target_height / (img_h*1.0f));
    int new_unpad_w = int(round(img_w * r));
    int new_unpad_h = int(round(img_h * r));
    int dw = target_width - new_unpad_w;
    int dh = target_height - new_unpad_h;
    dw /= 2;
    dh /= 2;
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_unpad_w, new_unpad_h));
    cv::Mat out;
    cv::copyMakeBorder(resized, out, dh, target_height - new_unpad_h - dh, dw, target_width - new_unpad_w - dw, cv::BORDER_CONSTANT, color);
    // trans: scale_x, scale_y, pad_x, pad_y
    trans[0] = r;
    trans[1] = r;
    trans[2] = dw;
    trans[3] = dh;
    return out;
}

std::vector<Segmentation> YOLOSeg::detect(const cv::Mat& image, float conf_threshold, float mask_threshold) {
    if (!session_) {
        std::cerr << "YOLOSeg not initialized!" << std::endl;
        return {};
    }
    cv::Vec4d trans;
    cv::Mat padded = letterbox(image, net_width_, net_height_, trans);
    cv::Mat blob = cv::dnn::blobFromImage(padded, 1.0/255.0, cv::Size(net_width_, net_height_), cv::Scalar(0, 0, 0), true, false);
    std::vector<int64_t> input_shape = {1, 3, net_width_, net_height_};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
        (float*)blob.data, 3 * net_width_ * net_height_, 
        input_shape.data(), input_shape.size());
    auto start = std::chrono::high_resolution_clock::now();
    auto outputs = session_->Run(Ort::RunOptions{nullptr},
        input_names_.data(), &input_tensor, 1, 
        output_names_.data(), output_names_.size());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Inference time: " << duration << " ms" << std::endl;
    float* output0_data = outputs[0].GetTensorMutableData<float>();
    auto output0_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    cv::Mat output0 = cv::Mat(cv::Size((int)output0_shape[2], (int)output0_shape[1]), 
                              CV_32F, output0_data).t();
    auto output1_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
    std::vector<int> mask_shape = {1, (int)output1_shape[1], (int)output1_shape[2], (int)output1_shape[3]};
    cv::Mat output1 = cv::Mat(mask_shape, CV_32F, outputs[1].GetTensorMutableData<float>());
    // 用letterbox参数修正ImageInfo
    ImageInfo img_info;
    img_info.raw_size = image.size();
    img_info.trans = cv::Vec4d(trans[0], trans[1], trans[2], trans[3]);
    std::vector<Segmentation> results;
    decode_output(output0, output1, img_info, results, conf_threshold, mask_threshold);
    return results;
}

void YOLOSeg::draw_result(cv::Mat& image, const std::vector<Segmentation>& results) {
    cv::Mat mask_overlay = image.clone();
    
    for (const auto& result : results) {
        int left = result.bbox.x;
        int top = result.bbox.y;
        
        // Get color for this class
        cv::Scalar color = get_random_color(result.id);
        
        // Draw bounding box
        cv::rectangle(image, result.bbox, color, 2);
        
        // Apply mask overlay
        if (result.mask.rows > 0 && result.mask.cols > 0) {
            mask_overlay(result.bbox).setTo(color, result.mask);
        }
        
        // Draw label
        std::string label = class_names_[result.id] + ":" + std::to_string(result.confidence).substr(0, 4);
        cv::putText(image, label, cv::Point(left, top - 10), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
    }
    
    // Blend mask with original image
    cv::addWeighted(image, 0.6, mask_overlay, 0.4, 0, image);
}

cv::Scalar YOLOSeg::get_random_color(int class_id) {
    static std::vector<cv::Scalar> colors;
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(128, 255);
    
    if (colors.empty()) {
        colors.resize(class_names_.size());
        for (int i = 0; i < class_names_.size(); ++i) {
            colors[i] = cv::Scalar(dis(gen), dis(gen), dis(gen));
        }
    }
    
    return colors[class_id % colors.size()];
}

bool YOLOSeg::process_video(const std::string& video_path, const std::string& output_path) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << video_path << std::endl;
        return false;
    }
    
    cv::VideoWriter writer;
    if (!output_path.empty()) {
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        cv::Size frame_size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        writer.open(output_path, fourcc, cap.get(cv::CAP_PROP_FPS), frame_size);
        if (!writer.isOpened()) {
            std::cerr << "Failed to create output video writer" << std::endl;
            return false;
        }
    }
    
    cv::Mat frame;
    int frame_count = 0;
    
    while (cap.read(frame)) {
        frame_count++;
        std::cout << "Processing frame " << frame_count << std::endl;
        
        // Detect objects
        auto results = detect(frame);
        
        // Draw results
        draw_result(frame, results);
        
        // Show frame (resize to fit screen while maintaining aspect ratio)
        cv::Mat display_frame;
        double scale = std::min(1280.0 / frame.cols, 720.0 / frame.rows);
        cv::Size display_size(frame.cols * scale, frame.rows * scale);
        cv::resize(frame, display_frame, display_size);
        cv::imshow("YOLOSeg Detection", display_frame);
        
        // Write to output video if specified
        if (!output_path.empty()) {
            writer.write(frame);
        }
        
        // Break on 'q' key
        char key = cv::waitKey(1) & 0xFF;
        if (key == 'q') {
            break;
        }
    }
    
    cap.release();
    if (!output_path.empty()) {
        writer.release();
    }
    cv::destroyAllWindows();
    
    return true;
}

bool YOLOSeg::process_image(const std::string& image_path, const std::string& output_path) {
    // Load image
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return false;
    }
    
    std::cout << "Processing image: " << image_path << std::endl;
    std::cout << "Image size: " << image.cols << "x" << image.rows << std::endl;
    
    // Detect objects
    auto results = detect(image);
    
    std::cout << "Detected " << results.size() << " objects" << std::endl;
    
    // Draw results
    draw_result(image, results);
    
    // Save result
    std::string final_output_path = output_path;
    if (final_output_path.empty()) {
        // Generate default output path
        size_t dot_pos = image_path.find_last_of('.');
        std::string base_name = image_path.substr(0, dot_pos);
        final_output_path = base_name + "_seg_result.jpg";
    }
    
    if (cv::imwrite(final_output_path, image)) {
        std::cout << "Result saved to: " << final_output_path << std::endl;
    } else {
        std::cerr << "Failed to save result to: " << final_output_path << std::endl;
        return false;
    }
    
    // Show result (resize to fit screen while maintaining aspect ratio)
    cv::Mat display_image;
    double scale = std::min(1280.0 / image.cols, 720.0 / image.rows);
    cv::Size display_size(image.cols * scale, image.rows * scale);
    cv::resize(image, display_image, display_size);
    cv::imshow("YOLOSeg Image Result", display_image);
    std::cout << "Press any key to close..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    return true;
} 
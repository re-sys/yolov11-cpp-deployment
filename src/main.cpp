#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <algorithm>
#include "ia/YOLO11.hpp"
#include "ia/YOLOSeg.hpp"

struct Args {
    std::string model_path = "../chpt/yolov11_seg.onnx";
    std::string classes_path = "./classes.txt";
    std::string input_path = "./input.mov";
    std::string output_path = "./output.mp4";
    bool use_gpu = false;
    float conf_threshold = 0.25f;
    float iou_threshold = 0.45f;
    float mask_threshold = 0.5f;
    bool segmentation = false;  // 新增：是否启用分割模式
    bool help = false;
};

void print_help() {
    std::cout << "YOLOv11 C++ Object Detection & Segmentation\n\n";
    std::cout << "Usage: ./yolov11_detector [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --model <path>      Path to ONNX model file (default: ./best_fixed.onnx)\n";
    std::cout << "  --classes <path>    Path to class names file (default: ./classes.txt)\n";
    std::cout << "  --input <path>      Path to input video file or camera device index (default: ./input.mov)\n";
    std::cout << "  --output <path>     Path to output video file (default: ./output.mp4)\n";
    std::cout << "  --gpu               Use GPU acceleration if available (default: false)\n";
    std::cout << "  --conf <value>      Confidence threshold (default: 0.25)\n";
    std::cout << "  --iou <value>       IoU threshold for NMS (default: 0.45)\n";
    std::cout << "  --mask <value>      Mask threshold for segmentation (default: 0.5)\n";
    std::cout << "  --seg               Enable instance segmentation mode (default: false)\n";
    std::cout << "  --help              Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  ./yolov11_detector --input=test_video.mp4 --output=result.mp4 --conf=0.3\n";
    std::cout << "  ./yolov11_detector --input=0 --gpu --seg  # Use webcam with GPU and segmentation\n";
    std::cout << "  ./yolov11_detector --input=test.jpg --seg --mask=0.3  # Image segmentation\n";
}

Args parse_args(int argc, char* argv[]) {
    Args args;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        
        if (arg == "--help" || arg == "-h") {
            args.help = true;
        } else if (arg.find("--model=") == 0) {
            args.model_path = arg.substr(8);
        } else if (arg.find("--classes=") == 0) {
            args.classes_path = arg.substr(10);
        } else if (arg.find("--input=") == 0) {
            args.input_path = arg.substr(8);
        } else if (arg.find("--output=") == 0) {
            args.output_path = arg.substr(9);
        } else if (arg == "--gpu") {
            args.use_gpu = true;
        } else if (arg.find("--conf=") == 0) {
            args.conf_threshold = std::stof(arg.substr(7));
        } else if (arg.find("--iou=") == 0) {
            args.iou_threshold = std::stof(arg.substr(6));
        } else if (arg.find("--mask=") == 0) {
            args.mask_threshold = std::stof(arg.substr(7));
        } else if (arg == "--seg") {
            args.segmentation = true;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
        }
    }
    
    return args;
}

bool is_camera_input(const std::string& input) {
    try {
        std::stoi(input);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool is_image_file(const std::string& input) {
    std::string lower = input;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    return lower.find(".jpg") != std::string::npos || 
           lower.find(".jpeg") != std::string::npos || 
           lower.find(".png") != std::string::npos || 
           lower.find(".bmp") != std::string::npos;
}

void draw_fps(cv::Mat& frame, double fps) {
    std::string fps_text = "FPS: " + std::to_string(static_cast<int>(fps));
    cv::putText(frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
}

int main(int argc, char* argv[]) {
    Args args = parse_args(argc, argv);
    
    if (args.help) {
        print_help();
        return 0;
    }
    
    std::cout << "YOLOv11 C++ Object Detection & Segmentation\n";
    std::cout << "============================================\n";
    std::cout << "Model: " << args.model_path << "\n";
    std::cout << "Classes: " << args.classes_path << "\n";
    std::cout << "Input: " << args.input_path << "\n";
    std::cout << "Output: " << args.output_path << "\n";
    std::cout << "GPU: " << (args.use_gpu ? "enabled" : "disabled") << "\n";
    std::cout << "Mode: " << (args.segmentation ? "Segmentation" : "Detection") << "\n";
    std::cout << "Confidence threshold: " << args.conf_threshold << "\n";
    std::cout << "IoU threshold: " << args.iou_threshold << "\n";
    if (args.segmentation) {
        std::cout << "Mask threshold: " << args.mask_threshold << "\n";
    }
    std::cout << "\n";
    
    // Initialize detector based on mode
    if (args.segmentation) {
        // Use YOLOSeg for segmentation
        YOLOSeg detector;
        if (!detector.initialize(args.model_path, args.classes_path, args.use_gpu)) {
            std::cerr << "Failed to initialize YOLOSeg detector!" << std::endl;
            return -1;
        }
        std::cout << "Using YOLOSeg for segmentation" << std::endl;
        
        // 如果是图像文件，进行单张图像处理
        if (is_image_file(args.input_path)) {
            std::string output_path = args.output_path;
            if (output_path == "./output.mp4") {
                // 如果是默认输出路径，改为图像输出
                size_t dot_pos = args.input_path.find_last_of('.');
                std::string base_name = args.input_path.substr(0, dot_pos);
                output_path = base_name + "_seg_result.jpg";
            }
            
            if (detector.process_image(args.input_path, output_path)) {
                std::cout << "Image processing completed. Output saved to: " << output_path << std::endl;
            } else {
                std::cerr << "Failed to process image!" << std::endl;
                return -1;
            }
            return 0;
        }
        
        // 视频处理
        if (!is_camera_input(args.input_path)) {
            // 对于视频文件，使用process_video方法
            std::string output_path = args.output_path;
            if (output_path == "./output.mp4") {
                size_t dot_pos = args.input_path.find_last_of('.');
                std::string base_name = args.input_path.substr(0, dot_pos);
                output_path = base_name + "_seg_result.mp4";
            }
            
            std::cout << "Processing video with YOLOSeg..." << std::endl;
            if (detector.process_video(args.input_path, output_path)) {
                std::cout << "Video processing completed. Output saved to: " << output_path << std::endl;
            } else {
                std::cerr << "Failed to process video!" << std::endl;
                return -1;
            }
            return 0;
        } else {
            // 摄像头处理
            int camera_id = std::stoi(args.input_path);
            cv::VideoCapture cap(camera_id);
            if (!cap.isOpened()) {
                std::cerr << "Error: Cannot open camera: " << camera_id << std::endl;
                return -1;
            }
            
            cv::Mat frame;
            while (cap.read(frame)) {
                auto results = detector.detect(frame, args.conf_threshold, args.mask_threshold);
                detector.draw_result(frame, results);
                
                cv::resize(frame, frame, cv::Size(640, 640));
                cv::imshow("YOLOSeg Camera", frame);
                
                char key = cv::waitKey(1) & 0xFF;
                if (key == 'q' || key == 27) {
                    break;
                }
            }
            
            cap.release();
            cv::destroyAllWindows();
            return 0;
        }
    } else {
        // Use YOLO11 for detection
        YOLO11 detector;
        if (!detector.initialize(args.model_path, args.classes_path, args.use_gpu)) {
            std::cerr << "Failed to initialize YOLO11 detector!" << std::endl;
            return -1;
        }
        
        std::cout << "Using YOLO11 for detection" << std::endl;
    
            // 如果是图像文件，进行单张图像处理
        if (is_image_file(args.input_path)) {
            std::cout << "Processing single image: " << args.input_path << std::endl;
            
            cv::Mat image = cv::imread(args.input_path);
            if (image.empty()) {
                std::cerr << "Error: Cannot read image file: " << args.input_path << std::endl;
                return -1;
            }
            
            cv::Mat result = image.clone();
            
            // 检测模式
            auto start_time = std::chrono::high_resolution_clock::now();
            auto detections = detector.detect(image, args.conf_threshold, args.iou_threshold);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            std::cout << "Detection completed in " << duration.count() << "ms" << std::endl;
            std::cout << "Detected " << detections.size() << " objects" << std::endl;
            
            detector.draw_detections(result, detections);
            
            // 保存结果
            std::string output_path = args.output_path;
            if (output_path == "./output.mp4") {
                // 如果是默认输出路径，改为图像输出
                size_t dot_pos = args.input_path.find_last_of('.');
                std::string base_name = args.input_path.substr(0, dot_pos);
                output_path = base_name + "_result.jpg";
            }
            
            cv::imwrite(output_path, result);
            std::cout << "Result saved to: " << output_path << std::endl;
            
            // 显示结果 (resize to fit screen while maintaining aspect ratio)
            cv::Mat display_result;
            double scale = std::min(1280.0 / result.cols, 720.0 / result.rows);
            cv::Size display_size(result.cols * scale, result.rows * scale);
            cv::resize(result, display_result, display_size);
            cv::imshow("YOLOv11 Result", display_result);
            cv::waitKey(0);
            
            return 0;
        }
    
    // 视频处理
    cv::VideoCapture cap;
    if (is_camera_input(args.input_path)) {
        int camera_id = std::stoi(args.input_path);
        cap.open(camera_id);
        std::cout << "Opening camera " << camera_id << std::endl;
    } else {
        cap.open(args.input_path);
        std::cout << "Opening video file: " << args.input_path << std::endl;
    }
    
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open input source: " << args.input_path << std::endl;
        return -1;
    }
    
    // Get video properties
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30.0;  // Default FPS for camera
    
    std::cout << "Video properties: " << frame_width << "x" << frame_height << " @ " << fps << " FPS\n\n";
    
    // Setup video writer for output
    cv::VideoWriter writer;
    if (!is_camera_input(args.input_path)) {
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        writer.open(args.output_path, fourcc, fps, cv::Size(frame_width, frame_height));
        
        if (!writer.isOpened()) {
            std::cerr << "Error: Cannot open output video file: " << args.output_path << std::endl;
            return -1;
        }
        std::cout << "Output will be saved to: " << args.output_path << std::endl;
    }
    
    // Performance tracking
    auto start_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    double avg_fps = 0.0;
    
    cv::Mat frame;
    std::cout << "\nProcessing... Press 'q' to quit.\n\n";
    
    while (true) {
        auto frame_start = std::chrono::high_resolution_clock::now();
        
        if (!cap.read(frame)) {
            if (is_camera_input(args.input_path)) {
                std::cerr << "Error reading from camera" << std::endl;
                break;
            } else {
                std::cout << "End of video file reached" << std::endl;
                break;
            }
        }
        
        cv::Mat result = frame.clone();
        
        // 检测模式
        auto detections = detector.detect(frame, args.conf_threshold, args.iou_threshold);
        detector.draw_detections(result, detections);
        
        // Calculate and display FPS
        auto frame_end = std::chrono::high_resolution_clock::now();
        auto frame_duration = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - frame_start);
        double current_fps = 1000.0 / frame_duration.count();
        
        frame_count++;
        avg_fps = (avg_fps * (frame_count - 1) + current_fps) / frame_count;
        
        draw_fps(result, current_fps);
        
        // Show frame (for camera input or debugging)
        if (is_camera_input(args.input_path)) {
            cv::imshow("YOLOv11 Detection", result);
            
            char key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) {  // 'q' or ESC
                break;
            }
        }
        
        // Write frame to output video
        if (writer.isOpened()) {
            writer.write(result);
        }
        
        // Print progress for video files
        if (!is_camera_input(args.input_path) && frame_count % 30 == 0) {
            std::cout << "Processed " << frame_count << " frames, Average FPS: " 
                      << static_cast<int>(avg_fps) << std::endl;
        }
    }
    
    // Cleanup
    cap.release();
    if (writer.isOpened()) {
        writer.release();
    }
    cv::destroyAllWindows();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    std::cout << "\nProcessing completed!\n";
    std::cout << "Total frames processed: " << frame_count << std::endl;
    std::cout << "Total time: " << total_duration.count() << " seconds\n";
    std::cout << "Average FPS: " << static_cast<int>(avg_fps) << std::endl;
    
    }
    
    return 0;
} 
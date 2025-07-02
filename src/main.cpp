#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "ia/YOLO11.hpp"

struct Args {
    std::string model_path = "./best.onnx";
    std::string classes_path = "./classes.txt";
    std::string input_path = "./input.mov";
    std::string output_path = "./output.mp4";
    bool use_gpu = false;
    float conf_threshold = 0.25f;
    float iou_threshold = 0.45f;
    bool help = false;
};

void print_help() {
    std::cout << "YOLOv11 C++ Object Detection\n\n";
    std::cout << "Usage: ./yolov11_detector [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --model <path>      Path to ONNX model file (default: ./best.onnx)\n";
    std::cout << "  --classes <path>    Path to class names file (default: ./classes.txt)\n";
    std::cout << "  --input <path>      Path to input video file or camera device index (default: ./input.mov)\n";
    std::cout << "  --output <path>     Path to output video file (default: ./output.mp4)\n";
    std::cout << "  --gpu               Use GPU acceleration if available (default: false)\n";
    std::cout << "  --conf <value>      Confidence threshold (default: 0.25)\n";
    std::cout << "  --iou <value>       IoU threshold for NMS (default: 0.45)\n";
    std::cout << "  --help              Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  ./yolov11_detector --input=test_video.mp4 --output=result.mp4 --conf=0.3\n";
    std::cout << "  ./yolov11_detector --input=0 --gpu  # Use webcam with GPU\n";
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
    
    std::cout << "YOLOv11 C++ Object Detection\n";
    std::cout << "============================\n";
    std::cout << "Model: " << args.model_path << "\n";
    std::cout << "Classes: " << args.classes_path << "\n";
    std::cout << "Input: " << args.input_path << "\n";
    std::cout << "Output: " << args.output_path << "\n";
    std::cout << "GPU: " << (args.use_gpu ? "enabled" : "disabled") << "\n";
    std::cout << "Confidence threshold: " << args.conf_threshold << "\n";
    std::cout << "IoU threshold: " << args.iou_threshold << "\n\n";
    
    // Initialize YOLO11
    YOLO11 detector;
    if (!detector.initialize(args.model_path, args.classes_path, args.use_gpu)) {
        std::cerr << "Failed to initialize YOLO11 detector!" << std::endl;
        return -1;
    }
    
    // Open input (video file or camera)
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
        
        // Perform detection
        auto detections = detector.detect(frame, args.conf_threshold, args.iou_threshold);
        
        // Draw detections
        detector.draw_detections(frame, detections);
        
        // Calculate and display FPS
        auto frame_end = std::chrono::high_resolution_clock::now();
        auto frame_duration = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - frame_start);
        double current_fps = 1000.0 / frame_duration.count();
        
        frame_count++;
        avg_fps = (avg_fps * (frame_count - 1) + current_fps) / frame_count;
        
        draw_fps(frame, current_fps);
        
        // Show frame (for camera input or debugging)
        if (is_camera_input(args.input_path)) {
            cv::imshow("YOLOv11 Detection", frame);
            
            char key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) {  // 'q' or ESC
                break;
            }
        }
        
        // Write frame to output video
        if (writer.isOpened()) {
            writer.write(frame);
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
    
    // Final statistics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    std::cout << "\nProcessing completed!\n";
    std::cout << "===================\n";
    std::cout << "Total frames processed: " << frame_count << "\n";
    std::cout << "Total time: " << total_duration.count() << " seconds\n";
    std::cout << "Average FPS: " << static_cast<int>(avg_fps) << "\n";
    
    if (!is_camera_input(args.input_path) && writer.isOpened()) {
        std::cout << "Output saved to: " << args.output_path << "\n";
    }
    
    return 0;
} 
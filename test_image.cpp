#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/ia/YOLO11.hpp"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: ./test_image <image_path>" << std::endl;
        return -1;
    }
    
    std::string image_path = argv[1];
    
    // Load image
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Error: Cannot load image " << image_path << std::endl;
        return -1;
    }
    
    std::cout << "Loaded image: " << image.cols << "x" << image.rows << std::endl;
    
    // Initialize YOLO11
    YOLO11 detector;
    if (!detector.initialize("./best.onnx", "./classes.txt", false)) {
        std::cerr << "Failed to initialize YOLO11 detector!" << std::endl;
        return -1;
    }
    
    std::cout << "YOLO11 initialized successfully" << std::endl;
    
    // Perform detection
    auto start = std::chrono::high_resolution_clock::now();
    auto detections = detector.detect(image, 0.25f, 0.45f);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Detection took: " << duration.count() << " ms" << std::endl;
    std::cout << "Found " << detections.size() << " objects" << std::endl;
    
    // Draw detections
    detector.draw_detections(image, detections);
    
    // Save result
    std::string output_path = "result_" + image_path;
    cv::imwrite(output_path, image);
    std::cout << "Result saved to: " << output_path << std::endl;
    
    // Display image
    cv::imshow("YOLOv11 Detection Result", image);
    std::cout << "Press any key to close..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    return 0;
} 
#include <iostream>
#include <onnxruntime_cxx_api.h>

int main() {
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ModelInfo");
        Ort::SessionOptions session_options;
        Ort::Session session(env, "./best_fixed.onnx", session_options);
        
        std::cout << "=== Model Information ===" << std::endl;
        
        // Print input information
        size_t num_input_nodes = session.GetInputCount();
        std::cout << "Number of inputs: " << num_input_nodes << std::endl;
        
        Ort::AllocatorWithDefaultOptions allocator;
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = session.GetInputNameAllocated(i, allocator);
            auto input_type_info = session.GetInputTypeInfo(i);
            auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
            auto input_shape = input_tensor_info.GetShape();
            
            std::cout << "Input " << i << ":" << std::endl;
            std::cout << "  Name: " << input_name.get() << std::endl;
            std::cout << "  Shape: [";
            for (size_t j = 0; j < input_shape.size(); j++) {
                std::cout << input_shape[j];
                if (j < input_shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        
        // Print output information
        size_t num_output_nodes = session.GetOutputCount();
        std::cout << "\nNumber of outputs: " << num_output_nodes << std::endl;
        
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = session.GetOutputNameAllocated(i, allocator);
            auto output_type_info = session.GetOutputTypeInfo(i);
            auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
            auto output_shape = output_tensor_info.GetShape();
            
            std::cout << "Output " << i << ":" << std::endl;
            std::cout << "  Name: " << output_name.get() << std::endl;
            std::cout << "  Shape: [";
            for (size_t j = 0; j < output_shape.size(); j++) {
                std::cout << output_shape[j];
                if (j < output_shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
} 
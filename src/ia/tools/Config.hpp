#pragma once

// Model configuration constants
#define DEFAULT_INPUT_WIDTH 640
#define DEFAULT_INPUT_HEIGHT 640
#define DEFAULT_CONF_THRESHOLD 0.25f
#define DEFAULT_IOU_THRESHOLD 0.45f

// Performance tuning constants
#ifdef DEBUG
    #define DEBUG_PRINT(x) std::cout << "[DEBUG] " << x << std::endl
    #define START_TIMER(name) auto start_##name = std::chrono::high_resolution_clock::now()
    #define END_TIMER(name) auto end_##name = std::chrono::high_resolution_clock::now(); \
                            auto duration_##name = std::chrono::duration_cast<std::chrono::milliseconds>(end_##name - start_##name); \
                            std::cout << "[TIMER] " << #name << ": " << duration_##name.count() << " ms" << std::endl
#else
    #define DEBUG_PRINT(x)
    #define START_TIMER(name)
    #define END_TIMER(name)
#endif

// Drawing configuration
#define BOX_COLOR_R 0
#define BOX_COLOR_G 255
#define BOX_COLOR_B 0
#define BOX_ALPHA 0.3f 
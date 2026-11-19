cmake_minimum_required(VERSION 3.16)
project(yolo_onnx_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)   # benchmarks in Debug are meaningless
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs dnn)

# Path to the unpacked ONNX Runtime release (contains include/ and lib/).
set(ONNXRUNTIME_DIR "" CACHE PATH "ONNX Runtime root directory")

add_executable(yolo_infer
  src/main.cpp
  src/detector.cpp
  src/opencv_detector.cpp)
target_include_directories(yolo_infer PRIVATE include)
target_link_libraries(yolo_infer PRIVATE ${OpenCV_LIBS})

if(ONNXRUNTIME_DIR)
  find_library(ORT_LIB onnxruntime PATHS ${ONNXRUNTIME_DIR}/lib REQUIRED NO_DEFAULT_PATH)
  target_sources(yolo_infer PRIVATE src/ort_detector.cpp)
  target_include_directories(yolo_infer PRIVATE ${ONNXRUNTIME_DIR}/include)
  target_link_libraries(yolo_infer PRIVATE ${ORT_LIB})
  target_compile_definitions(yolo_infer PRIVATE WITH_ORT)
  if(WIN32)
    # Copy onnxruntime.dll next to the exe so it is found at runtime.
    add_custom_command(TARGET yolo_infer POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              ${ONNXRUNTIME_DIR}/lib/onnxruntime.dll $<TARGET_FILE_DIR:yolo_infer>)
  else()
    set_target_properties(yolo_infer PROPERTIES BUILD_RPATH ${ONNXRUNTIME_DIR}/lib)
  endif()
  message(STATUS "ONNX Runtime backend: ON (${ONNXRUNTIME_DIR})")
else()
  message(STATUS "ONNX Runtime backend: OFF (pass -DONNXRUNTIME_DIR=...)")
endif()

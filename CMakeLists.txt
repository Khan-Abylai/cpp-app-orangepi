cmake_minimum_required(VERSION 3.25)
project(orangepiVideoDecoder)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
find_package(OpenCV REQUIRED)
find_package(PkgConfig)
find_package(nlohmann_json REQUIRED)
find_package(CURL REQUIRED)
find_package(ncnn REQUIRED)
#if(NOT ncnn_FOUND)
#    set(ncnn_DIR "/tmp/ncnn/build/install/lib/cmake/ncnn" CACHE PATH "Path to ncnn CMake configuration directory")
#    find_package(ncnn REQUIRED PATHS "${ncnn_DIR}")
#endif()

pkg_check_modules(GST REQUIRED gstreamer-1.0)
pkg_check_modules(GST_RTP REQUIRED gstreamer-rtp-1.0)
pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0)
pkg_check_modules(LIBONNXRUNTIME REQUIRED IMPORTED_TARGET libonnxruntime)

include_directories(${CURL_INCLUDE_DIR})
include_directories(${GST_INCLUDE_DIRS} ${GST_APP_INCLUDE_DIRS} ${GST_RTP_INCLUDE_DIRS})
include_directories(${LIBONNXRUNTIME_INCLUDE_DIRS})

find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /tmp/onnxruntime/include/onnxruntime/core/session/)
find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /tmp/onnxruntime/build/Linux/MinSizeRel)

if(NOT ONNX_RUNTIME_SESSION_INCLUDE_DIRS)
    set(ONNX_RUNTIME_SESSION_INCLUDE_DIRS /home/orangepi/onnxruntime/include/onnxruntime/core/session/)
endif()

if(NOT ONNX_RUNTIME_LIB)
    find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /home/orangepi/onnxruntime/build/Linux/MinSizeRel)
endif()

add_executable(orangepiVideoDecoder src/app/Counter.h src/ParkingApp.cpp src/SharedQueue.h src/ILogger.cpp src/IThreadLauncher.h src/ITimer.h src/Profile.h src/RandomStringGenerator.cpp src/RandomStringGenerator.h src/ILogger.h src/client/CameraClientLauncher.cpp src/client/FrameData.cpp src/client/FrameData.cpp src/client/CameraScope.cpp src/client/GstreamerReader.cpp src/app/DetectionService.cpp src/app/DetectionService.h src/app/DetectionService.h src/app/DetectionService.cpp src/Config.cpp src/Config.h src/app/Utils.cpp src/app/Utils.h src/app/Constants.h src/app/LicensePlate.h src/app/LicensePlate.cpp src/app/DetectionMobilenet.cpp src/app/DetectionMobilenet.h src/app/Detection.cpp src/app/Detection.h src/app/TemplateMatching.cpp src/app/TemplateMatching.h src/app/LPRecognizer.h src/app/LPRecognizer.cpp src/app/Car.h src/app/Car.cpp src/app/BaseCarTracker.h src/app/BaseCarTracker.cpp src/app/Package.h src/app/Package.cpp src/app/CalibParams.h src/app/CalibParamsUpdater.cpp src/app/CalibParamsUpdater.h src/app/CalibParams.cpp src/app/MaskCarTracker.cpp src/app/MaskCarTracker.h src/app/LPRecognizerService.cpp src/app/LPRecognizerService.h src/app/PackageSender.h src/app/PackageSender.cpp src/app/Snapshot.h src/app/Snapshot.cpp src/app/SnapshotSender.h src/app/SnapshotSender.cpp src/app/DetectionNCNN.cpp)

target_include_directories(orangepiVideoDecoder PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${ncnn_INCLUDE_DIRS})

target_link_libraries(orangepiVideoDecoder PRIVATE ${ONNX_RUNTIME_LIB} ${OpenCV_LIBS} ${GST_LIBRARIES} ${GST_APP_LIBRARIES} ${GST_RTP_LIBRARIES} nlohmann_json::nlohmann_json ${CURL_LIBRARIES} /usr/local/lib/libcpr.so ${ALL_INTERFACE_TARGET} ncnn)


target_compile_definitions(orangepiVideoDecoder PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:_SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING>
        )
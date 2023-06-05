//
// Created by kartykbayev on 12/28/22.
//

#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <utility>
#include <vector>
#include "LicensePlate.h"
#include "Constants.h"

class DetectionMobilenet {
public:

    DetectionMobilenet();

    std::vector<std::shared_ptr<LicensePlate>> detect(const cv::Mat &frame);

private:
    struct Point {
        float _x;
        float _y;
    };
    struct bbox {
        float x1;
        float y1;
        float x2;
        float y2;
        float s;
        Point point[5];
    };

    struct box {
        float cx;
        float cy;
        float sx;
        float sy;
    };


    void create_anchor(std::vector<box> &anchor, int w, int h);

    Ort::Env env{nullptr};
    Ort::SessionOptions sessionOptions{nullptr};
    Ort::Session session{nullptr};
    std::vector<const char *> inputNames;
    std::vector<const char *> outputNames;
    bool isDynamicInputShape{};
    std::vector<std::vector<int>> min_sizes = {{16,  32},
                                               {64,  128},
                                               {256, 512}};

    int steps[3] = {8, 16, 32};
    const int IMG_W = 640, IMG_H = 480, IMG_C=3, BBOX_COUNT=4, KEYPOINT_COUNT=10;
    const float DET_THRESHOLD = 0.7, NMS_THRESHOLD = 0.4;
    const std::vector<float> variances{0.1, 0.2};

    std::vector<box> anchor;

    std::vector<float> prepareImage(const cv::Mat &frame);

    void nms(std::vector<bbox> &input_boxes, float NMS_THRESH);

    static inline bool cmp(bbox a, bbox b);


};

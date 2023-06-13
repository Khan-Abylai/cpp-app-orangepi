#pragma once

#include <opencv2/opencv.hpp>
#include <utility>
#include <vector>
#include "LicensePlate.h"
#include "Constants.h"
#include "net.h"
using namespace std;
class DetectionNCNN {
public:
    explicit DetectionNCNN(const ncnn::Extractor& ex);

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


    std::vector<std::vector<int>> min_sizes = {{16,  32},
                                               {64,  128},
                                               {256, 512}};

    int steps[3] = {8, 16, 32};
    const int IMG_W = 640, IMG_H = 480, IMG_C = 3, BBOX_COUNT = 4, KEYPOINT_COUNT = 10;
    const float DET_THRESHOLD = 0.7, NMS_THRESHOLD = 0.4;
    const std::vector<float> variances{0.1, 0.2};

    std::vector<box> anchor;

    void nms(std::vector<bbox> &input_boxes, float NMS_THRESH);
    std::vector<float> makeFlattened(ncnn::Mat &val);
    static inline bool cmp(bbox a, bbox b);

    ncnn::Extractor ex;
};



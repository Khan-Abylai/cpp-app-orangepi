//
// Created by kartykbayev on 8/28/22.
//
#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <utility>
#include <vector>
#include "LicensePlate.h"
#include "Constants.h"

class Detection {
public:
    Detection();

    std::vector<std::shared_ptr<LicensePlate>> detect(const cv::Mat &frame);
    std::shared_ptr<LicensePlate> chooseOneLicensePlate(std::vector<std::shared_ptr<LicensePlate>> &licensePlates);
private:
    Ort::Env env{nullptr};
    Ort::SessionOptions sessionOptions{nullptr};
    Ort::Session session{nullptr};

    std::shared_ptr<LicensePlate> getMaxAreaPlate(std::vector<std::shared_ptr<LicensePlate>> &licensePlates);

    std::vector<const char *> inputNames;
    std::vector<const char *> outputNames;
    bool isDynamicInputShape{};
    cv::Size2f inputImageShape;

    std::vector<float> prepareImage(const cv::Mat &frame);


    std::vector<std::shared_ptr<LicensePlate>>
    nms(const std::vector<std::tuple<float, std::shared_ptr<LicensePlate>>> &licensePlates);

    float iou(const std::shared_ptr<LicensePlate> &firstLp, const std::shared_ptr<LicensePlate> &secondLp);

    std::vector<std::tuple<float, std::shared_ptr<LicensePlate>>>
    getLicensePlates(std::vector<float> lpPredictions, int frameWidth, int frameHeight);
};


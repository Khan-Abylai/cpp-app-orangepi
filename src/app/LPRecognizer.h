
#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <utility>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <array>
#include <filesystem>
#include "Constants.h"
#include "Utils.h"
#include "TemplateMatching.h"
#include "LicensePlate.h"
#include <cmath>

class LPRecognizer {
public:
    LPRecognizer();

    std::vector<std::pair<std::string, float>> predict(const std::vector<cv::Mat> &frames);

private:
    Ort::Env env{nullptr};
    Ort::SessionOptions sessionOptions{nullptr};
    Ort::Session session{nullptr};

    std::vector<const char *> inputNames;
    std::vector<const char *> outputNames;
    float RECOGNIZER_THRESHOLD = 0.85;

    std::vector<float> prepareImages(const std::vector<cv::Mat> &frames);

    const std::string
            ALPHABET = "-0123456789abcdefghijklmnopqrstuvwxyz",
            NETWORK_INPUT_NAME = "INPUT",
            NETWORK_DIM_NAME = "DIMENSIONS",
            NETWORK_OUTPUT_NAME = "OUTPUT";

    const int
            SEQUENCE_SIZE = 30,
            ALPHABET_SIZE = 37,
            BLANK_INDEX = 0,
            IMG_WIDTH = Constants::RECT_LP_W,
            IMG_HEIGHT = Constants::RECT_LP_H,
            IMG_CHANNELS = Constants::IMG_CHANNELS,
            INPUT_SIZE = IMG_CHANNELS * IMG_HEIGHT * IMG_WIDTH,
            OUTPUT_SIZE = SEQUENCE_SIZE * ALPHABET_SIZE,
            MAX_BATCH_SIZE = Constants::RECOGNIZER_MAX_BATCH_SIZE,
            MAX_PLATE_SIZE = 12;

    std::vector<float> softmax(std::vector<float> &score_vec);

    cv::Mat whiteImage = cv::Mat(Constants::RECT_LP_H, Constants::RECT_LP_W, CV_8UC3, cv::Scalar(255, 255, 255));
    std::unique_ptr<TemplateMatching> templateMatching;

    std::pair<cv::Mat, cv::Mat> divideImageIntoHalf(const cv::Mat &lpImage);
};


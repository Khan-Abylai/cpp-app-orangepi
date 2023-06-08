#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>
#include "client/CameraScope.h"
#include "app/Utils.h"

class Config {
public:

    static bool parseJson(const std::string &fileName);

    static const std::vector<CameraScope> &getCameras();

    static const int &getTimeSentSnapshots();

    static const float &getRecognizerThreshold();

    static const float &getDetectorThreshold();

    static std::pair<float, float> getCalibrationSizes();

};
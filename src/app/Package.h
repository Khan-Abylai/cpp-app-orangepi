#pragma once

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

#include "Utils.h"
#include "LicensePlate.h"
#include "../ITimer.h"
#include "../Config.h"

class Package : public ITimer {
public:
    Package(std::string cameraIp, std::string licensePlateLabel,
            cv::Mat carImage, cv::Mat plateImage, std::string strBoundingBox,
            std::basic_string<char> direction, std::string resultSendUrlParam);

    std::string getPackageJsonString() const;

    std::string getDisplayPackageJsonString() const;

    const std::string &getPlateLabel() const;

    const std::string &getCameraIp() const;

    static std::string convertBoundingBoxToStr(const std::shared_ptr<LicensePlate> &licensePlate);

    std::string getDirection() const;

    const std::string &getResultSendUrl() const;

private:
    std::string resultSendUrl;
    std::string direction;
    std::string carModel;
    std::string cameraIp;
    time_t eventTime;
    std::string licensePlateLabel;
    cv::Mat carImage;
    cv::Mat plateImage;
    std::string strBoundingBox;
    float detProb, recProb;
};

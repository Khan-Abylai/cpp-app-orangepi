#include "Package.h"

#include <utility>

using namespace std;
using json = nlohmann::json;

Package::Package(string cameraIp, string licensePlateLabel,
                 cv::Mat carImage, cv::Mat plateImage,
                 string strBoundingBox, basic_string<char> direction, std::string resultSendUrlParam)
        : cameraIp(std::move(cameraIp)), licensePlateLabel(std::move(licensePlateLabel)),
          carImage(std::move(carImage)), plateImage(std::move(plateImage)),
          strBoundingBox(std::move(strBoundingBox)), direction(std::move(direction)),
          resultSendUrl{std::move(resultSendUrlParam)} {
    eventTime = time_t(nullptr);
};

string Package::convertBoundingBoxToStr(const shared_ptr<LicensePlate> &licensePlate) {
    auto frameSize = licensePlate->getCarImageSize();
    auto frameWidth = (float) frameSize.width;
    auto frameHeight = (float) frameSize.height;
    return Utils::pointToStr(licensePlate->getLeftTop().x / frameWidth,
                             licensePlate->getLeftTop().y / frameHeight) + ", " +
           Utils::pointToStr(licensePlate->getLeftBottom().x / frameWidth,
                             licensePlate->getLeftBottom().y / frameHeight) + ", " +
           Utils::pointToStr(licensePlate->getRightTop().x / frameWidth,
                             licensePlate->getRightTop().y / frameHeight) + ", " +
           Utils::pointToStr(licensePlate->getRightBottom().x / frameWidth,
                             licensePlate->getRightBottom().y / frameHeight);
}


string Package::getPackageJsonString() const {
    json packageJson;
    packageJson["ip_address"] = cameraIp;
    packageJson["event_time"] = Utils::dateTimeToStr(eventTime);
    packageJson["car_number"] = licensePlateLabel;
    packageJson["car_picture"] = Utils::encodeImgToBase64(carImage);
    packageJson["lp_picture"] = Utils::encodeImgToBase64(plateImage);
    packageJson["lp_rect"] = strBoundingBox;
    packageJson["car_model"] = carModel;
    packageJson["direction"] = "forward";

    return packageJson.dump();
}

const string &Package::getPlateLabel() const {
    return licensePlateLabel;
}

const string &Package::getCameraIp() const {
    return cameraIp;
}

string Package::getDirection() const {
    return direction;
}

const std::string &Package::getResultSendUrl() const {
    return resultSendUrl;
}




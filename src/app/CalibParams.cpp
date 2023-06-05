#include "CalibParams.h"

using namespace std;
using json = nlohmann::json;

CalibParams::CalibParams(const string &serverIp, const string &cameraIp,
                         pair<float, float> calibrationSizes)
        : ILogger(
        "Calib Params " + cameraIp), cameraIp(cameraIp) {

    FRAME_WIDTH_HD = calibrationSizes.first;
    FRAME_HEIGHT_HD = calibrationSizes.second;
    calibParamsUrl = serverIp + cameraIp;
    LOG_INFO("Calibration Parameter Url : %s", calibParamsUrl.c_str());
    getMask();
}

void CalibParams::getMask() {
    auto responseText = sendRequestForMaskPoints();
    auto polygonPoints = getPolygonPoints(responseText, "mask2");
    auto subPolygonPoints = getPolygonPoints(responseText, "mask");

    if (responseText.empty() || responseText.length() <= 2) {
        minWidth = INT_MIN;
        minHeight = INT_MIN;
        maxHeight = INT_MAX;
        maxWidth = INT_MAX;
    } else {
        minHeight = getDimension("height", "min", responseText);
        maxHeight = getDimension("height", "max", responseText);
        minWidth = getDimension("width", "min", responseText);
        maxWidth = getDimension("width", "max", responseText);
    }
    lock_guard<mutex> guard(maskAccessChangeMutex);
    mask = cv::Mat::zeros((int) FRAME_HEIGHT_HD, (int) FRAME_WIDTH_HD, CV_8UC1);
    mask2 = cv::Mat::zeros((int) FRAME_HEIGHT_HD, (int) FRAME_WIDTH_HD, CV_8UC1);

    cv::fillConvexPoly(mask, polygonPoints, WHITE_COLOR);
    cv::fillConvexPoly(mask2, subPolygonPoints, WHITE_COLOR);
}

vector<cv::Point2i> CalibParams::getPolygonPoints(const string &polygonPointsStr, const string &maskType) const {
    vector<cv::Point2i> polygonPoints;
    if (polygonPointsStr.empty() || polygonPointsStr.length() <= 2) {
        polygonPoints = getDefaultPolygonPoints();
    } else {
        auto polygonPointsJson = json::parse(polygonPointsStr);
        for (auto &point: polygonPointsJson[maskType])
            polygonPoints.emplace_back(point["x"].get<int>(), point["y"].get<int>());
        if (polygonPoints.empty())
            polygonPoints = getDefaultPolygonPoints();
    }
    return polygonPoints;
}

string CalibParams::sendRequestForMaskPoints() {
    cpr::Response response = cpr::Get(cpr::Url{calibParamsUrl}, cpr::VerifySsl(0));
    if (response.status_code >= 400 || response.status_code == 0) {
        LOG_ERROR("%s Error [%d] making request for mask", cameraIp.data(), response.status_code);
        return "";
    }
    return response.text;
}

cv::Point2i CalibParams::getRelatedPoint(const cv::Point2f &point, const cv::Size &imageSize) const {
    auto xPoint = (point.x * FRAME_WIDTH_HD) / imageSize.width;
    auto yPoint = (point.y * FRAME_HEIGHT_HD) / imageSize.height;
    return cv::Point2i{(int) xPoint, (int) yPoint};
}

bool CalibParams::isPointInTheMask(const cv::Point2i &point) {
    lock_guard<mutex> guard(maskAccessChangeMutex);
    return mask.at<uchar>(point.y, point.x) == WHITE_COLOR;
}

int CalibParams::getDimension(const std::string &dimension, const std::string &key, const std::string &parsedStr) {
    auto jsonVal = json::parse(parsedStr);
    return jsonVal[key][dimension].get<int>();
}

bool
CalibParams::isLicensePlateInSelectedArea(const shared_ptr<LicensePlate> &licensePlate, const std::string &maskType) {
    auto yPointInGround = projectYPointToGround(licensePlate);

    cv::Point2f centerPointInGround{static_cast<float>(licensePlate->getCenter().x), yPointInGround};

    auto centerPointHd = getRelatedPoint(centerPointInGround, licensePlate->getCarImageSize());
    if (maskType == "main")
        return isPointInTheMask(centerPointHd);
    else
        return isPointInTheSubMask(centerPointHd);
}

float CalibParams::projectYPointToGround(const shared_ptr<LicensePlate> &licensePlate) {
    return (float) (licensePlate->getCenter().y);
}

const string &CalibParams::getCameraIp() const {
    return cameraIp;
}

bool CalibParams::isPointInTheSubMask(const cv::Point2i &point) {
    lock_guard<mutex> guard(maskAccessChangeMutex);
    return mask2.at<uchar>(point.y, point.x) == WHITE_COLOR;
}

std::vector<cv::Point2i> CalibParams::getDefaultPolygonPoints() const {
    vector<cv::Point2i> polygonPoints;

    polygonPoints.emplace_back(0, 0);
    polygonPoints.emplace_back(static_cast<int>(FRAME_WIDTH_HD), 0);
    polygonPoints.emplace_back(static_cast<int>(FRAME_WIDTH_HD), static_cast<int>(FRAME_HEIGHT_HD));
    polygonPoints.emplace_back(0, static_cast<int>(FRAME_HEIGHT_HD));

    return polygonPoints;
}



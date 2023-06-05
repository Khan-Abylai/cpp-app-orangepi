#include "Car.h"

using namespace std;

Car::Car(const int &platesCount) : ILogger("Car "), MIN_NUM_PLATES_COLLECTED{platesCount} {
    direction = Directions::initial;
}

void Car::addLicensePlateToCount(shared_ptr<LicensePlate> licensePlate) {
    auto newLicensePlate = make_shared<Counter<shared_ptr<LicensePlate>>>(std::move(licensePlate));
    auto licensePlateIterator = licensePlates.find(newLicensePlate);
    if (licensePlateIterator != licensePlates.end()) {
        const auto &foundLicensePlate = *licensePlateIterator;
        foundLicensePlate->incrementOccurrence();
        foundLicensePlate->renewItem(newLicensePlate->getItem());
        if (!mostCommonPlate || mostCommonPlate->getNumberOfOccurrence() < foundLicensePlate->getNumberOfOccurrence())
            mostCommonPlate = foundLicensePlate;
    } else {
        if (!mostCommonPlate)
            mostCommonPlate = newLicensePlate;
        licensePlates.insert({newLicensePlate});
    }
}

bool Car::isTrackPoint(const cv::Point2f &centerPoint) {
    if (trackingPoints.empty())
        return true;

    auto distance = Utils::calculateDistanceBetweenTwoPoints(trackingPoints.back(), centerPoint);
    return distance >= MIN_DISTANCE_BETWEEN_POINTS;
}

bool Car::isLicensePlateBelongsToCar(const shared_ptr<LicensePlate> &curLicensePlate,
                                     double lastFrameRTPTimestamp) {
    double minDistance = INT_MAX;
    int trackingIndex;

    for (int i = (int) trackingPoints.size() - 1; i >= 0; i--) {
        auto distance = Utils::calculateDistanceBetweenTwoPoints(trackingPoints[i], curLicensePlate->getCenter());
        if (minDistance > distance) {
            minDistance = distance;
            trackingIndex = i;
            if (minDistance <= MIN_DISTANCE_BETWEEN_POINTS)
                break;
        }
    }
    double timeDiffBetweenFrames = curLicensePlate->getRTPtimestamp() - lastFrameRTPTimestamp;


    if (trackingPoints.size() - trackingIndex <= MAX_DIFFERENCE_BETWEEN_TRACKING_POINTS &&
        timeDiffBetweenFrames < MAX_DIFFERENCE_BETWEEN_TIMESTAMPS)
        return true;

    return isLicensePlateBelongsToCar(curLicensePlate->getPlateLabel());
}

void Car::addTrackingPoint(const cv::Point2f &point, bool packageSent) {
    if (isTrackPoint(point)) {
        trackingPoints.push_back(point);
        if (trackingPoints.size() == 1) {
            direction = Directions::initial;
        } else if (trackingPoints.size() % TRACK_POINTS_THRESHOLD == 0 && !packageSent) {
            int checker = 0;
            int rev_checker = 0;

            for (const auto &trackPoint: trackingPoints) {
                if (trackPoint.y <= point.y) checker++;
                if (trackPoint.y > point.y) rev_checker++;
            }

            if (checker >= 5) {
                direction = Directions::forward;
            } else if (rev_checker <= 5) {
                direction = Directions::reverse;
            }
        }
    }
}

bool Car::isLicensePlateBelongsToCar(const string &otherPlateLabel) {
    int distance = Utils::calculateEditDistance(mostCommonPlate->getItem()->getPlateLabel(), otherPlateLabel);
    return distance <= EDIT_DISTANCE_THRESHOLD;
}

const shared_ptr<LicensePlate> &Car::getMostCommonLicensePlate() const {
    LOG_INFO("---- found plates -----");
    for (const auto &plate: licensePlates) {
        LOG_INFO("plate: %s, count: %d", plate->getItem()->getPlateLabel().data(), plate->getNumberOfOccurrence());
    }
    return mostCommonPlate->getItem();
}

void Car::drawTrackingPoints(cv::Mat &image) {
    for (const auto &centerPoint: trackingPoints) {
        cv::circle(image, centerPoint, 2, cv::Scalar(0, 255, 0), cv::FILLED, cv::LINE_8);
    }
}

void Car::drawBoundingBoxPoints(cv::Mat &image, const shared_ptr<LicensePlate> &licensePlate) {
    cv::circle(image, licensePlate->getLeftTop(), 3, cv::Scalar(0, 255, 0));
    cv::circle(image, licensePlate->getLeftBottom(), 3, cv::Scalar(0, 255, 0));
    cv::circle(image, licensePlate->getRightTop(), 3, cv::Scalar(0, 255, 0));
    cv::circle(image, licensePlate->getRightBottom(), 3, cv::Scalar(0, 255, 0));
}

Directions Car::getDirection() {
    return direction;
}


bool Car::doesPlatesCollected() {
    int platesCollected = 0;
    for (const auto &plate: licensePlates) platesCollected += plate->getNumberOfOccurrence();

    if (platesCollected > MIN_NUM_PLATES_COLLECTED) return true;
    return false;
}

void Car::addTrackingCarModel(const basic_string<char> &carModel) {
    if (trackingCarModels.find(carModel) == trackingCarModels.end())trackingCarModels.insert(make_pair(carModel, 1));
    else trackingCarModels[carModel]++;
}


std::string Car::getCarModel() {
    if (findEntryWithLargestValue(trackingCarModels).first == trackingCarModels.end()->first) {
        return "Not recognized";
    } else {
        return findEntryWithLargestValue(trackingCarModels).first;
    }
}

std::pair<std::string, int> Car::getCarModelWithOccurrence() {
    if (findEntryWithLargestValue(trackingCarModels).first == trackingCarModels.end()->first) {
        return pair<std::string, int>{"Not Recognized", 0};
    } else {
        return pair<std::string, int>{findEntryWithLargestValue(trackingCarModels).first,
                                      findEntryWithLargestValue(trackingCarModels).second};
    }
}

bool Car::doesPlatesDoubleCollected() {
    int platesCollected = 0;
    for (const auto &plate: licensePlates) platesCollected += plate->getNumberOfOccurrence();

    return (int) MIN_NUM_PLATES_COLLECTED < platesCollected;
}

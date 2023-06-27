#include "MaskCarTracker.h"

using namespace std;

MaskCarTracker::MaskCarTracker(shared_ptr<SharedQueue<shared_ptr<Package>>> packageQueue,
                               const shared_ptr<CalibParams> &calibParams, bool useDirection, int platesCount,
                               int timeBetweenResendingPlates) : BaseCarTracker(std::move(packageQueue),
                                                                                calibParams->getCameraIp()),
                                                                 calibParams{(calibParams)}, platesCount{platesCount},
                                                                 timeBetweenResendingPlates{timeBetweenResendingPlates},
                                                                 USE_DIRECTION{useDirection} {
}

void MaskCarTracker::track(const shared_ptr<LicensePlate> &licensePlate) {


    if (!currentCar || !currentCar->isLicensePlateBelongsToCar(licensePlate, lastFrameRTPTimestamp)) {

        if (currentCar && !isPlateAlreadySent) {
            LOG_INFO("plate %s wasn't sent...", currentCar->getMostCommonLicensePlate()->getPlateLabel().data());
        }

        currentCar = createNewCar(platesCount);
        isPlateAlreadySent = false;

        LOG_INFO("tracking new car %s", licensePlate->getPlateLabel().data());
    }
    lastFrameRTPTimestamp = licensePlate->getRTPtimestamp();
    currentCar->addTrackingPoint(licensePlate->getCenter(), isPlateAlreadySent);

    currentCar->addLicensePlateToCount(licensePlate);

    considerToResendLP(licensePlate);

    if (isSufficientMomentToSendLP(licensePlate, "main") && !isPlateAlreadySent) {
        if ((currentCar->getDirection() == Directions::forward && currentCar->doesPlatesCollected()) ||
            (!USE_DIRECTION && currentCar->doesPlatesCollected())) {
            sendMostCommonPlate();
        }
    }

}

void MaskCarTracker::considerToResendLP(const shared_ptr<LicensePlate> &licensePlate) {
    if ((isPlateAlreadySent && isSufficientTimePassedToSendPlate() &&
         isSufficientMomentToSendLP(licensePlate, "resend") && currentCar->getDirection() == Directions::forward) ||
        (isPlateAlreadySent && isSufficientTimePassedToSendPlate() && !USE_DIRECTION)) {
        LOG_INFO("resending plate....");
        licensePlate->setDirection("forward");
        createAndPushPackage(licensePlate);
        lastTimeLPSent = lastFrameRTPTimestamp;
    }
}

bool MaskCarTracker::isSufficientTimePassedToSendPlate() {
    return lastFrameRTPTimestamp - lastTimeLPSent >= timeBetweenResendingPlates;
}

void MaskCarTracker::sendMostCommonPlate() {
    shared_ptr<LicensePlate> mostCommonLicensePlate = currentCar->getMostCommonLicensePlate();
    if (currentCar->getDirection() == Directions::forward) {
        mostCommonLicensePlate->setDirection(("forward"));
    } else if (currentCar->getDirection() == Directions::reverse) {
        mostCommonLicensePlate->setDirection("reverse");
    }
    createAndPushPackage(mostCommonLicensePlate);
    isPlateAlreadySent = true;
    lastTimeLPSent = lastFrameRTPTimestamp;
}

void MaskCarTracker::run() {
    calibParamsUpdater = make_unique<CalibParamsUpdater>(calibParams);
    calibParamsUpdater->run();
}

void MaskCarTracker::shutdown() {
    LOG_INFO("service is shutting down");
    calibParamsUpdater->shutdown();
}

void MaskCarTracker::saveFrame(const shared_ptr<LicensePlate> &licensePlate) {
    auto curPlate = currentCar->getMostCommonLicensePlate();
    cv::imwrite(Constants::IMAGE_DIRECTORY + licensePlate->getPlateLabel() + Constants::JPG_EXTENSION,
                licensePlate->getCarImage());
}

bool
MaskCarTracker::isSufficientMomentToSendLP(const shared_ptr<LicensePlate> &licensePlate, const string &typeOfSend) {
    if (typeOfSend == "main") {
        if (calibParams->isLicensePlateInSelectedArea(licensePlate, "sub") && currentCar->doesPlatesCollected()) {
            string model = "NotDefined";
            currentCar->addTrackingCarModel(model);
            return true;
        } else {
            return false;
        }
    } else {
        if (isPlateAlreadySent && isSufficientTimePassedToSendPlate()) {
            string model = "NotDefined";
            currentCar->addTrackingCarModel(model);
            return true;
        } else {
            return false;
        }
    }
}


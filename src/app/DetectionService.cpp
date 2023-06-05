//
// Created by kartykbayev on 6/1/23.
//

#include "DetectionService.h"

using namespace std;

void DetectionService::shutdown() {
    LOG_INFO("service is shutting down");
    shutdownFlag = true;
    frameQueue->push(nullptr);
}

DetectionService::DetectionService(std::shared_ptr<SharedQueue<std::unique_ptr<FrameData>>> frameQueue,
                                   const std::string &cameraIp,
                                   std::shared_ptr<LPRecognizerService> lpRecognizerService) : ILogger(
        "DetectionService"), frameQueue{std::move(frameQueue)}, licensePlateRecognizerService{
        std::move(lpRecognizerService)} {
    LOG_INFO("Camera ip: %s", cameraIp.c_str());
    this->detection = make_shared<Detection>();
}

void DetectionService::run() {
    while (!shutdownFlag) {
        auto frameData = frameQueue->wait_and_pop();
        if (frameData == nullptr) continue;
        auto frame = frameData->getFrame();

        auto startTime = chrono::high_resolution_clock::now();
        auto detectionResult = detection->detect(frame);
        auto endTime = chrono::high_resolution_clock::now();


        if (detectionResult.empty()) continue;

        auto licensePlate = chooseOneLicensePlate(detectionResult);
        licensePlate->setPlateImage(frame); // perspective transform and set to plateImage
        licensePlate->setCameraIp(frameData->getIp());
        licensePlate->setCarImage(std::move(frame));
        licensePlate->setRTPtimestamp(frameData->getRTPtimestamp());
        licensePlateRecognizerService->addToQueue(std::move(licensePlate));
    }
}

std::shared_ptr<LicensePlate>
DetectionService::getMaxAreaPlate(std::vector<std::shared_ptr<LicensePlate>> &licensePlates) {
    float maxArea = -1.0;
    shared_ptr<LicensePlate> licensePlate;
    for (auto &curPlate: licensePlates) {
        float area = curPlate->getArea();
        if (area > maxArea) {
            maxArea = area;
            licensePlate = std::move(curPlate);
        }
    }
    return licensePlate;
}

std::shared_ptr<LicensePlate>
DetectionService::chooseOneLicensePlate(std::vector<std::shared_ptr<LicensePlate>> &licensePlates) {
    shared_ptr<LicensePlate> licensePlate;

    if (licensePlates.size() > 1)  // if more than one license plate
        licensePlate = getMaxAreaPlate(licensePlates); // move licensePlate
    else
        licensePlate = std::move(licensePlates.front());

    return licensePlate;
}

//
// Created by kartykbayev on 6/1/23.
//
#pragma once


#include "../IThreadLauncher.h"
#include "../ILogger.h"
#include "../SharedQueue.h"
#include "../client/FrameData.h"
#include "Detection.h"

#include "DetectionMobilenet.h"
#include "LPRecognizerService.h"

class DetectionService : public IThreadLauncher, public ILogger {


public:
    explicit DetectionService(std::shared_ptr<SharedQueue<std::unique_ptr<FrameData>>> frameQueue,
                              const std::string &cameraIp, std::shared_ptr<LPRecognizerService> lpRecognizerService);

    void run() override;

    void shutdown() override;

private:
    time_t lastFrameRTPTimestamp = time(nullptr);
    time_t lastTimeSnapshotSent = time(nullptr);
    int timeBetweenSendingSnapshots = 1;

    std::shared_ptr<Detection> detection;
    std::shared_ptr<SharedQueue<std::unique_ptr<FrameData>>> frameQueue;
    std::shared_ptr<LPRecognizerService> licensePlateRecognizerService;

    static std::shared_ptr<LicensePlate> getMaxAreaPlate(std::vector<std::shared_ptr<LicensePlate>> &licensePlates);

    static std::shared_ptr<LicensePlate>
    chooseOneLicensePlate(std::vector<std::shared_ptr<LicensePlate>> &licensePlates);
};

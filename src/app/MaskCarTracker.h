#pragma once

#include <cmath>
#include "BaseCarTracker.h"
#include "../RandomStringGenerator.h"
#include "CalibParams.h"
#include "CalibParamsUpdater.h"

class MaskCarTracker : public BaseCarTracker {
public:
    explicit MaskCarTracker(std::shared_ptr<SharedQueue<std::shared_ptr<Package>>> packageQueue,
                            const std::shared_ptr<CalibParams> &calibParams,
                            bool useDirection,
                            int platesCount,
                            int timeBetweenResendingPlates);

    void track(const std::shared_ptr<LicensePlate> &licensePlate) override;

    void run() override;

    void shutdown() override;

private:
    bool USE_DIRECTION;
    int platesCount;
    int timeBetweenResendingPlates;

    std::shared_ptr<CalibParams> calibParams;
    std::unique_ptr<CalibParamsUpdater> calibParamsUpdater;

    bool isPlateAlreadySent{false};
    double lastTimeLPSent = 0;

    void sendMostCommonPlate() override;

    void saveFrame(const std::shared_ptr<LicensePlate> &licensePlate);

    bool isSufficientTimePassedToSendPlate();

    void considerToResendLP(const std::shared_ptr<LicensePlate> &licensePlate);

    bool isSufficientMomentToSendLP(const std::shared_ptr<LicensePlate> &licensePlate,
                                    const std::string &typeOfSend);

};

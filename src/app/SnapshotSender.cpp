//
// Created by kartykbayev on 8/22/22.
//

#include "SnapshotSender.h"

#include <utility>
#include <glib.h>

using namespace std;

SnapshotSender::SnapshotSender(
        std::shared_ptr<SharedQueue<std::shared_ptr<Snapshot>>> snapshotQueue,
        const std::vector<CameraScope> &cameraScope) : ILogger("Snapshot Sender -------------"),
                                                       snapshotQueue{std::move(snapshotQueue)} {
    for (const auto &cameraIp: cameraScope) {
        lastSendTimes.insert({cameraIp.getCameraIp(), time(nullptr)});
        LOG_INFO("Camera ip:%s will send snapshot to:%s", cameraIp.getCameraIp().c_str(),
                 cameraIp.getSnapshotSendUrl().c_str());
    }
}


void SnapshotSender::run() {
    time_t beginTime = time(nullptr);

    queue<cpr::AsyncResponse> responses;

    while (!shutdownFlag) {
        auto package = snapshotQueue->wait_and_pop();
        if (package == nullptr)
            continue;
        lastFrameRTPTimestamp = time(nullptr);
        if (lastFrameRTPTimestamp - lastTimeSnapshotSent >= 100) {
            lastTimeSnapshotSent = time(nullptr);
        }

        responses.push(sendRequests(package->getPackageJsonString(), package->getSnapshotUrl()));
        while (responses.size() > MAX_FUTURE_RESPONSES) {
            std::queue<cpr::AsyncResponse> empty;
            std::swap(responses, empty);
        }


    }
}

void SnapshotSender::shutdown() {
    LOG_INFO("service is shutting down");
    shutdownFlag = true;
    snapshotQueue->push(nullptr);
}

cpr::AsyncResponse SnapshotSender::sendRequests(const string &jsonString, const string &serverUrl) {
    return cpr::PostAsync(cpr::Url{serverUrl}, cpr::VerifySsl(false), cpr::Body{jsonString},
                          cpr::Timeout{SEND_REQUEST_TIMEOUT}, cpr::Header{{"Content-Type", "application/json"}});
}

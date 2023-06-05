#include "PackageSender.h"

#include <utility>

using namespace std;


PackageSender::PackageSender(shared_ptr<SharedQueue<shared_ptr<Package>>> packageQueue,
                             const vector<CameraScope> &cameraScope)
        : ILogger("Package Sender --------------------"),
          packageQueue{std::move(packageQueue)} {

    for (const auto &camera: cameraScope) {
        LOG_INFO("Camera ip:%s will send packages to the: %s",
                 camera.getCameraIp().c_str(),
                 camera.getResultSendURL().c_str());
    }

    for (const auto &camera: cameraScope)
        lastSendTimes.insert({camera.getCameraIp(), time(nullptr)});
}

void PackageSender::run() {
    time_t beginTime = time(nullptr);

    queue<future<cpr::Response>> futureResponses;
    queue<cpr::AsyncResponse> responses;


    while (!shutdownFlag) {
        auto package = packageQueue->wait_and_pop();
        if (package == nullptr) continue;

        LOG_INFO("%s %s %s", package->getCameraIp().data(), package->getPlateLabel().data(),
                 Utils::dateTimeToStr(time_t(nullptr)).c_str());

        responses.push(sendRequests(package->getPackageJsonString(), package->getResultSendUrl()));


        while (responses.size() > MAX_FUTURE_RESPONSES) {
            responses.pop();
        }
    }
}

void PackageSender::shutdown() {
    LOG_INFO("service is shutting down");
    shutdownFlag = true;
    packageQueue->push(nullptr);
}

cpr::AsyncResponse PackageSender::sendRequests(const string &jsonString, const std::string &serverUrl) {
    return cpr::PostAsync(
            cpr::Url{serverUrl},
            cpr::VerifySsl(false),
            cpr::Body{jsonString},
            cpr::Timeout{SEND_REQUEST_TIMEOUT},
            cpr::Header{{"Content-Type", "application/json"}});
}



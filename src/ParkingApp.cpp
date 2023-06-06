#include <iostream>
#include <thread>
#include <chrono>
#include <csignal>
#include <condition_variable>
#include <mutex>
#include <atomic>
#include "client/FrameData.h"
#include "SharedQueue.h"
#include "IThreadLauncher.h"
#include "client/CameraClientLauncher.h"
#include "app/DetectionService.h"
#include "app/LPRecognizerService.h"
#include "Config.h"
#include "app/Package.h"
#include "app/PackageSender.h"
#include "app/Snapshot.h"
#include "app/SnapshotSender.h"

using namespace std;

atomic<bool> shutdownFlag = false;
condition_variable shutdownEvent;
mutex shutdownMutex;

void signalHandler(int signum) {
    cout << "signal is to shutdown" << endl;
    shutdownFlag = true;
    shutdownEvent.notify_all();
}


//TODO: finalize functionality of snapshot sending

int main(int argc, char *argv[]) {

    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    signal(SIGKILL, signalHandler);
    signal(SIGHUP, signalHandler);
    signal(SIGABRT, signalHandler);

    string configFileName;

    if (argc <= 1)
        configFileName = "config.json";
    else
        configFileName = argv[1];

    if (!Config::parseJson(configFileName))
        return -1;
    vector<shared_ptr<IThreadLauncher>> services;
    auto snapshotQueue = make_shared<SharedQueue<shared_ptr<Snapshot>>>();
    auto packageQueue = make_shared<SharedQueue<shared_ptr<Package>>>();
    vector<shared_ptr<SharedQueue<unique_ptr<FrameData>>>> frameQueues;

    auto cameras = Config::getCameras();

    auto lpRecognizerService = make_shared<LPRecognizerService>(packageQueue, cameras, Config::getRecognizerThreshold(),
                                                                Config::getCalibrationSizes());
    services.emplace_back(lpRecognizerService);

    for (const auto &camera: cameras) {
        auto frameQueue = make_shared<SharedQueue<unique_ptr<FrameData>>>();
        auto detectionService = make_shared<DetectionService>(frameQueue, snapshotQueue, camera.getCameraIp(),
                                                              lpRecognizerService);
        frameQueues.push_back(std::move(frameQueue));
        services.emplace_back(detectionService);
    }

    shared_ptr<IThreadLauncher> clientStarter;
    clientStarter = make_shared<CameraClientLauncher>(cameras, frameQueues,
                                                      false);
    services.emplace_back(clientStarter);

    auto packageSender = make_shared<PackageSender>(packageQueue, Config::getCameras());
    services.emplace_back(packageSender);

    auto snapshotSender = make_shared<SnapshotSender>(snapshotQueue, Config::getCameras());
    services.emplace_back(snapshotSender);

    vector<thread> threads;
    for (const auto &service: services) {
        threads.emplace_back(&IThreadLauncher::run, service);
    }

    unique_lock<mutex> shutdownLock(shutdownMutex);
    while (!shutdownFlag) {
        auto timeout = chrono::hours(24);
        if (shutdownEvent.wait_for(shutdownLock, timeout, [] { return shutdownFlag.load(); })) {
            cout << "shutting all services" << endl;
        }
    }

    for (int i = 0; i < services.size(); i++) {
        services[i]->shutdown();
        if (threads[i].joinable())
            threads[i].join();
    }

    std::cout << "Hello, World!" << std::endl;
    return 0;
}

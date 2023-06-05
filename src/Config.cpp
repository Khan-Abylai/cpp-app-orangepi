#include "Config.h"
#include "client/CameraScope.h"

using namespace std;
using json = nlohmann::json;

std::vector<CameraScope> cameras;
int timeBetweenSnapshotSend = 1;
float recognizerThreshold = 0.95;
float detectorThreshold = 0.85;
float calibrationWidth = 1920;
float calibrationHeight = 1080;

bool Config::parseJson(const string &fileName) {
    try {
        ifstream configFile(fileName);
        if (!configFile.is_open())
            throw runtime_error("Config file not found");

        json configs = json::parse(configFile);

        if (configs.find("cameras") == configs.end())
            throw runtime_error("Camera IP Entities not defined");
        auto cameraEntities = configs["cameras"];


        for (const auto &cameraEntity: cameraEntities) {
            std::string nodeIp, snapshotSendIp, resultSendIp;
            bool use_direction;
            int time_between_sent_plates, plates_count;

            if (cameraEntity.find("node_ip") == cameraEntity.end())
                throw runtime_error("node ip not defined");
            nodeIp = cameraEntity["node_ip"].get<string>();

            if (cameraEntity.find("result_send_endpoint") == cameraEntity.end())
                throw runtime_error("result_send_endpoint not defined");
            else
                resultSendIp = cameraEntity["result_send_endpoint"].get<string>();


            if (cameraEntity.find("snapshot_send_ip") == cameraEntity.end())
                throw runtime_error("Snapshot url not defined");
            else
                snapshotSendIp = cameraEntity["snapshot_send_ip"].get<string>();

            if (cameraEntity.find("use_direction") == cameraEntity.end() ||
                cameraEntity["use_direction"].get<int>() == 0)
                use_direction = 0;
            else
                use_direction = 1;

            if (cameraEntity.find("plates_count") == cameraEntity.end())
                plates_count = 3;
            else
                plates_count = cameraEntity["plates_count"].get<int>();

            if (cameraEntity.find("time_between_sent_plates") == cameraEntity.end())
                time_between_sent_plates = 3;
            else
                time_between_sent_plates = cameraEntity["time_between_sent_plates"].get<int>();

            auto rtspUrl = cameraEntity["rtsp_url"].get<string>();
            auto cameraIp = cameraEntity["camera_ip"].get<string>();

            auto cameraScope = CameraScope(cameraIp, rtspUrl,
                                           use_direction,
                                           plates_count, time_between_sent_plates, resultSendIp, nodeIp,
                                           snapshotSendIp);
            cameras.push_back(cameraScope);
        }

        if (configs.find("calibration_width") == configs.end())
            calibrationWidth = 1920.0;
        else
            calibrationWidth = configs["calibration_width"].get<float>();

        if (configs.find("calibration_height") == configs.end())
            calibrationHeight = 1080.0;
        else
            calibrationHeight = configs["calibration_height"].get<float>();

        if (configs.find("time_between_snapshot_send") == configs.end())
            timeBetweenSnapshotSend = 1;
        else
            timeBetweenSnapshotSend = configs["time_between_snapshot_send"].get<int>();

        if (configs.find("rec_threshold") == configs.end())
            recognizerThreshold = 0.95;
        else
            recognizerThreshold = configs["rec_threshold"].get<float>();

        if (configs.find("det_threshold") == configs.end())
            detectorThreshold = 0.8;
        else
            detectorThreshold = configs["det_threshold"].get<float>();

    } catch (exception &e) {
        cout << "Exception occurred during config parse: " << e.what() << endl;
        return false;
    }
    return true;
}

const std::vector<CameraScope> &Config::getCameras() {
    return cameras;
}


const float &Config::getRecognizerThreshold() {
    return recognizerThreshold;
}

const float &Config::getDetectorThreshold() {
    return detectorThreshold;
}

const int &Config::getTimeSentSnapshots() {
    return timeBetweenSnapshotSend;
}

std::pair<float, float> Config::getCalibrationSizes() {
    return std::pair<float, float>{calibrationWidth, calibrationHeight};
}

























//
// Created by Камалхан Артыкбаев on 14.10.2022.
//
#pragma once

#include <regex>
#include <string>

class CameraScope {
public:
    CameraScope(std::string &cameraIp, std::string &rtspUrl, bool isUseDirection,
                int platesCollectionCount,
                int resendingInterval,
                std::string &resultSendURL, std::string &nodeIp,
                std::string &snapshotUrl);

    const std::string &getCameraIp() const;

    const std::string &getCameraRTSPUrl() const;

    const std::string &getResultSendURL() const;

    const std::string &getNodeIP() const;

    const std::string &getSnapshotSendUrl() const;

    bool useDirection() const;

    int platesCount() const;

    int timeBetweenSentPlates() const;


private:
    std::string CAMERA_IP, RTSP_URL, RESULT_SEND_URL, NODE_IP, SNAPSHOT_SEND_IP;
    bool USE_DIRECTION;
    int PLATES_COUNT, TIME_BETWEEN_SENT_PLATES;
};
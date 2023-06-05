//
// Created by Камалхан Артыкбаев on 14.10.2022.
//

#include "CameraScope.h"

CameraScope::CameraScope(std::string &cameraIp, std::string &rtspUrl, bool isUseDirection,
                         int platesCollectionCount,
                         int resendingInterval,
                         std::string &resultSendURL, std::string &nodeIp,
                         std::string &snapshotUrl) {
    this->CAMERA_IP = std::move(cameraIp);
    this->RTSP_URL = std::move(rtspUrl);
    this->TIME_BETWEEN_SENT_PLATES = resendingInterval;
    this->PLATES_COUNT = platesCollectionCount;
    this->USE_DIRECTION = isUseDirection;
    this->RESULT_SEND_URL = std::move(resultSendURL);
    this->NODE_IP = std::move(nodeIp);
    this->SNAPSHOT_SEND_IP = std::move(snapshotUrl);
}

const std::string &CameraScope::getCameraIp() const {
    return CAMERA_IP;
}

const std::string &CameraScope::getCameraRTSPUrl() const {
    return RTSP_URL;
}

bool CameraScope::useDirection() const {
    return USE_DIRECTION;
}

int CameraScope::platesCount() const {
    return PLATES_COUNT;
}

int CameraScope::timeBetweenSentPlates() const {
    return TIME_BETWEEN_SENT_PLATES;
}

const std::string &CameraScope::getResultSendURL() const {
    return RESULT_SEND_URL;
}

const std::string &CameraScope::getNodeIP() const {
    return NODE_IP;
}

const std::string &CameraScope::getSnapshotSendUrl() const {
    return SNAPSHOT_SEND_IP;
}


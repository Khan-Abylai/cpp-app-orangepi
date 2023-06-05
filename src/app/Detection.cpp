//
// Created by kartykbayev on 8/28/22.
//

#include "Detection.h"

using namespace std;

Detection::Detection() {
    env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "ONNX_DETECTION");
    sessionOptions = Ort::SessionOptions();

    session = Ort::Session(env, Constants::detectorModelFilepath.c_str(), sessionOptions);
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    std::vector<int64_t> inputTensorShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();

    this->isDynamicInputShape = false;


    inputNames.push_back(session.GetInputName(0, allocator));
    outputNames.push_back(session.GetOutputName(0, allocator));

    this->inputImageShape = cv::Size2f(cv::Size(512, 512));
}

std::vector<std::shared_ptr<LicensePlate>> Detection::detect(const cv::Mat &frame) {
    float *blob = nullptr;
    std::vector<int64_t> inputTensorShape{1, 3, 512, 512};

    cv::Mat copyImage;
    frame.copyTo(copyImage);
    auto flattenImage = prepareImage(frame);

    size_t inputTensorSize = Utils::vectorProduct(inputTensorShape);


    std::vector<float> inputTensorValues(inputTensorSize);

    for (int64_t i = 0; i < 1; ++i) {
        std::copy(flattenImage.begin(),
                  flattenImage.end(),
                  inputTensorValues.begin() + i * inputTensorSize / 1);
    }

    std::vector<Ort::Value> inputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensorValues.data(), inputTensorSize,
            inputTensorShape.data(), inputTensorShape.size()
    ));

    auto *inputTen = inputTensors[0].GetTensorData<float>();
    std::vector<int64_t> inShape = inputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t count = inputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> input(inputTen, inputTen + count);


    std::vector<Ort::Value> outputTensors = this->session.Run(Ort::RunOptions{nullptr},
                                                              inputNames.data(),
                                                              inputTensors.data(),
                                                              1,
                                                              outputNames.data(),
                                                              1);

    auto *rawOutput = outputTensors[0].GetTensorData<float>();
    std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t count2 = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> output(rawOutput, rawOutput + count2);

    if (output.empty()) return std::vector<std::shared_ptr<LicensePlate >>{};
    vector<tuple<float, shared_ptr<LicensePlate>>> licensePlatesWithProbs = getLicensePlates(move(output),
                                                                                             copyImage.cols,
                                                                                             copyImage.rows);
    sort(licensePlatesWithProbs.begin(), licensePlatesWithProbs.end(), greater<>());
    return nms(licensePlatesWithProbs);
}

std::vector<float> Detection::prepareImage(const cv::Mat &frame) {
    vector<float> flattenedImage;
    int IMG_WIDTH = 512, IMG_HEIGHT = 512, INPUT_SIZE = 3 * IMG_WIDTH * IMG_HEIGHT;
    flattenedImage.resize(INPUT_SIZE, 0);
    cv::Mat resizedFrame;
    resize(frame, resizedFrame, cv::Size(IMG_WIDTH, IMG_HEIGHT));
    for (int row = 0; row < resizedFrame.rows; row++) {
        for (int col = 0; col < resizedFrame.cols; col++) {
            uchar *pixels = resizedFrame.data + resizedFrame.step[0] * row + resizedFrame.step[1] * col;
            flattenedImage[row * IMG_WIDTH + col] =
                    static_cast<float>(2 * (pixels[0] / Constants::PIXEL_MAX_VALUE - 0.5));

            flattenedImage[row * IMG_WIDTH + col + IMG_HEIGHT * IMG_WIDTH] =
                    static_cast<float>(2 * (pixels[1] / Constants::PIXEL_MAX_VALUE - 0.5));

            flattenedImage[row * IMG_WIDTH + col + 2 * IMG_HEIGHT * IMG_WIDTH] =
                    static_cast<float>(2 * (pixels[2] / Constants::PIXEL_MAX_VALUE - 0.5));
        }
    }
    return move(flattenedImage);
}

std::vector<std::tuple<float, std::shared_ptr<LicensePlate>>>
Detection::getLicensePlates(std::vector<float> lpPredictions, int frameWidth, int frameHeight) {
    vector<tuple<float, shared_ptr<LicensePlate>>> licensePlatesWithProbs;
    const int PLATE_GRID_SIZE = 16,
            IMG_WIDTH = 512,
            IMG_HEIGHT = 512,
            PLATE_COORDINATE_SIZE = 13,
            PLATE_GRID_WIDTH = IMG_WIDTH / PLATE_GRID_SIZE,
            PLATE_GRID_HEIGHT = IMG_HEIGHT / PLATE_GRID_SIZE,
            PLATE_OUTPUT_SIZE = PLATE_COORDINATE_SIZE * PLATE_GRID_HEIGHT * PLATE_GRID_WIDTH;

    const float LP_NMS_THRESHOLD = 0.4;
    const float LP_PROB_THRESHOLD = 0.8;

    float scaleWidth = static_cast<float>(frameWidth) / IMG_WIDTH;
    float scaleHeight = static_cast<float>(frameHeight) / IMG_HEIGHT;

    for (int index = 0; index < PLATE_GRID_HEIGHT * PLATE_GRID_WIDTH; ++index) {
        int current_index = index * PLATE_COORDINATE_SIZE;

        int prob_index = current_index + 12;
        float prob = lpPredictions[prob_index];
        if (prob > LP_PROB_THRESHOLD) {
            float x = lpPredictions[current_index] * scaleWidth;
            float y = lpPredictions[current_index + 1] * scaleHeight;
            float x1 = lpPredictions[current_index + 4] * scaleWidth + x;
            float y1 = lpPredictions[current_index + 5] * scaleHeight + y;

            float x2 = lpPredictions[current_index + 6] * scaleWidth + x;
            float y2 = lpPredictions[current_index + 7] * scaleHeight + y;

            float x3 = lpPredictions[current_index + 8] * scaleWidth + x;
            float y3 = lpPredictions[current_index + 9] * scaleHeight + y;

            float x4 = lpPredictions[current_index + 10] * scaleWidth + x;
            float y4 = lpPredictions[current_index + 11] * scaleHeight + y;

            int new_h = static_cast<int>(((y2 - y1) + (y4 - y3)) / 2);
            int new_w = static_cast<int>(((x3 - x1) + (x4 - x2)) / 2);
            licensePlatesWithProbs.emplace_back(
                    make_tuple(prob, make_shared<LicensePlate>(static_cast<int>(x), static_cast<int>(y),
                                                               abs(new_w), abs(new_h),
                                                               static_cast<int>(floor(x1)),
                                                               static_cast<int>(floor(y1)),
                                                               static_cast<int>(floor(x2)),
                                                               static_cast<int>(ceil(y2)),
                                                               static_cast<int>(ceil(x3)),
                                                               static_cast<int>(floor(y3)),
                                                               static_cast<int>(ceil(x4)),
                                                               static_cast<int>(ceil(y4)))));
        }

    }


    return move(licensePlatesWithProbs);
}

std::vector<std::shared_ptr<LicensePlate>>
Detection::nms(const vector<std::tuple<float, std::shared_ptr<LicensePlate>>> &licensePlates) {
    const float LP_NMS_THRESHOLD = 0.4;
    const float LP_PROB_THRESHOLD = 0.8;

    vector<shared_ptr<LicensePlate>> filteredLicensePlates;
    vector<bool> isFiltered;
    isFiltered.reserve(licensePlates.size());
    for (int lpIndex = 0; lpIndex < licensePlates.size(); lpIndex++) {
        isFiltered[lpIndex] = false;
    }

    for (int lpIndex = 0; lpIndex < licensePlates.size(); lpIndex++) {
        if (isFiltered[lpIndex]) continue;

        isFiltered[lpIndex] = true;
        auto [_, licensePlate] = licensePlates[lpIndex];

        for (int filterLpIndex = lpIndex + 1; filterLpIndex < licensePlates.size(); filterLpIndex++) {
            auto &[_, anotherLicensePlate] = licensePlates[filterLpIndex];
            if (iou(licensePlate, anotherLicensePlate) > LP_NMS_THRESHOLD)
                isFiltered[filterLpIndex] = true;
        }

        filteredLicensePlates.emplace_back(move(licensePlate));
    }
    sort(filteredLicensePlates.begin(), filteredLicensePlates.end(),
         [](const shared_ptr<LicensePlate> &a, const shared_ptr<LicensePlate> &b) {
             return (a->getCenter().y == b->getCenter().y) ? (a->getCenter().x < b->getCenter().x)
                                                           : (a->getCenter().y < b->getCenter().y);
         });

    return move(filteredLicensePlates);
}

float Detection::iou(const shared_ptr<LicensePlate> &firstLp, const shared_ptr<LicensePlate> &secondLp) {
    float firstLpArea =
            (firstLp->getRightBottom().x - firstLp->getLeftTop().x + 1) *
            (firstLp->getRightBottom().y - firstLp->getLeftTop().y + 1);
    float secondLpArea =
            (secondLp->getRightBottom().x - secondLp->getLeftTop().x + 1) *
            (secondLp->getRightBottom().y - secondLp->getLeftTop().y + 1);

    float intersectionX2 = min(firstLp->getRightBottom().x, secondLp->getRightBottom().x);
    float intersectionY2 = min(firstLp->getRightBottom().y, secondLp->getRightBottom().y);
    float intersectionX1 = max(firstLp->getLeftTop().x, secondLp->getLeftTop().x);
    float intersectionY1 = max(firstLp->getLeftTop().y, secondLp->getLeftTop().y);

    float intersectionX = (intersectionX2 - intersectionX1 + 1);
    float intersectionY = (intersectionY2 - intersectionY1 + 1);

    if (intersectionX < 0)
        intersectionX = 0;

    if (intersectionY < 0)
        intersectionY = 0;

    float intersectionArea = intersectionX * intersectionY;

    return intersectionArea / (firstLpArea + secondLpArea - intersectionArea);
}

std::shared_ptr<LicensePlate> Detection::chooseOneLicensePlate(vector<std::shared_ptr<LicensePlate>> &licensePlates) {
    shared_ptr<LicensePlate> licensePlate;

    if (licensePlates.size() > 1)  // if more than one license plate
        licensePlate = getMaxAreaPlate(licensePlates); // move licensePlate
    else
        licensePlate = move(licensePlates.front());

    return licensePlate;
}

std::shared_ptr<LicensePlate> Detection::getMaxAreaPlate(vector<std::shared_ptr<LicensePlate>> &licensePlates) {
    float maxArea = -1.0;
    shared_ptr<LicensePlate> licensePlate;
    for (auto &curPlate: licensePlates) {
        float area = curPlate->getArea();
        if (area > maxArea) {
            maxArea = area;
            licensePlate = move(curPlate);
        }
    }
    return licensePlate;
}

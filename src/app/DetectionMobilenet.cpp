//
// Created by kartykbayev on 12/28/22.
//

#include "DetectionMobilenet.h"

using namespace std;

DetectionMobilenet::DetectionMobilenet() {
    env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "ONNX_DETECTION");
    sessionOptions = Ort::SessionOptions();

    session = Ort::Session(env, Constants::detectorMobilenetModelFilepath.c_str(), sessionOptions);
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    std::vector<int64_t> inputTensorShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
    this->isDynamicInputShape = false;

    if (inputTensorShape[2] == -1 && inputTensorShape[3] == -1) {
        std::cout << "Dynamic input shape" << std::endl;
        this->isDynamicInputShape = true;
    }

    for (auto shape: inputTensorShape)
        std::cout << "Input shape: " << shape << std::endl;
    inputNames.push_back(session.GetInputName(0, allocator));
    outputNames.push_back(session.GetOutputName(0, allocator));
    outputNames.push_back(session.GetOutputName(1, allocator));
    outputNames.push_back(session.GetOutputName(2, allocator));
    create_anchor(anchor, IMG_W, IMG_H);

}

std::vector<std::shared_ptr<LicensePlate>> DetectionMobilenet::detect(const cv::Mat &frame) {
    std::vector<int64_t> inputTensorShape{1, IMG_C, IMG_H, IMG_W};
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
                                                              3);
    auto *rawOutput1 = outputTensors[0].GetTensorData<float>();
    std::vector<int64_t> outputShape1 = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t count1 = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> output1(rawOutput1, rawOutput1 + count1);

    auto *rawOutput2 = outputTensors[1].GetTensorData<float>();
    std::vector<int64_t> outputShape2 = outputTensors[1].GetTensorTypeAndShapeInfo().GetShape();
    size_t count2 = outputTensors[1].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> output2(rawOutput2, rawOutput2 + count2);

    auto *rawOutput3 = outputTensors[2].GetTensorData<float>();
    std::vector<int64_t> outputShape3 = outputTensors[2].GetTensorTypeAndShapeInfo().GetShape();
    size_t count3 = outputTensors[2].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> output3(rawOutput3, rawOutput3 + count3);

    std::vector<bbox> total_box;
    for (int i = 0; i < output2.size(); ++i) {
        if (i % 2 != 0 && output2[i] >= DET_THRESHOLD) {
            int real_index = (i - 1) / 2;
            int bbox_index = real_index * BBOX_COUNT;
            int landm_index = real_index * KEYPOINT_COUNT;
            box tmp = anchor[real_index];
            box tmp1;
            bbox result;

            tmp1.cx = tmp.cx + output1[bbox_index] * 0.1 * tmp.sx;
            tmp1.cy = tmp.cy + output1[bbox_index + 1] * 0.1 * tmp.sy;
            tmp1.sx = tmp.sx * exp(output1[bbox_index + 2] * 0.2);
            tmp1.sy = tmp.sy * exp(output1[bbox_index + 3] * 0.2);

            result.x1 = (tmp1.cx - tmp1.sx / 2) * IMG_W;
            if (result.x1 < 0)
                result.x1 = 0;
            result.y1 = (tmp1.cy - tmp1.sy / 2) * IMG_H;
            if (result.y1 < 0)
                result.y1 = 0;
            result.x2 = (tmp1.cx + tmp1.sx / 2) * IMG_W;
            if (result.x2 > 640)
                result.x2 = 640;
            result.y2 = (tmp1.cy + tmp1.sy / 2) * IMG_H;
            if (result.y2 > 480)
                result.y2 = 480;
            result.s = output2[i];

            for (int j = 0; j < KEYPOINT_COUNT / 2; ++j) {
                int indx = landm_index + j * 2;
                result.point[j]._x = (tmp.cx + output3[indx] * 0.1 * tmp.sx) * (float) IMG_W;
                result.point[j]._y = (tmp.cy + output3[indx + 1] * 0.1 * tmp.sy) * (float) IMG_H;
            }
            total_box.push_back(result);

        }
    }

    if (total_box.empty()) return {};

    std::sort(total_box.begin(), total_box.end(), cmp);
    nms(total_box, NMS_THRESHOLD);
    std::vector<bbox> boxes;
    for (int j = 0; j < total_box.size(); ++j) {
        boxes.push_back(total_box[j]);
    }

    float sc_w = (float) copyImage.cols / (float) IMG_W;
    float sc_h = (float) copyImage.rows / (float) IMG_H;


    auto final_lp_keypoints = boxes[0];
    float LT_x = final_lp_keypoints.point[0]._x * sc_w;
    float LT_y = final_lp_keypoints.point[0]._y * sc_h;
    float RT_x = final_lp_keypoints.point[1]._x * sc_w;
    float RT_y = final_lp_keypoints.point[1]._y * sc_h;
    float CP_x = final_lp_keypoints.point[2]._x * sc_w;
    float CP_y = final_lp_keypoints.point[2]._y * sc_h;
    float LB_x = final_lp_keypoints.point[3]._x * sc_w;
    float LB_y = final_lp_keypoints.point[3]._y * sc_h;
    float RB_x = final_lp_keypoints.point[4]._x * sc_w;
    float RB_y = final_lp_keypoints.point[4]._y * sc_h;
    float plate_width = ((RT_x - LT_x) + (RB_x - LB_x)) / 2;
    float plate_height = ((LB_y - LT_y) + (RB_y - RT_y)) / 2;
    auto lp = make_shared<LicensePlate>(static_cast<int>(CP_x), static_cast<int>(CP_y),
                                        static_cast<int>(plate_width), static_cast<int>(plate_height),
                                        static_cast<int>(floor(LT_x)),
                                        static_cast<int>(floor(LT_y)),
                                        static_cast<int>(floor(LB_x)),
                                        static_cast<int>(ceil(LB_y)),
                                        static_cast<int>(ceil(RT_x)),
                                        static_cast<int>(floor(RT_y)),
                                        static_cast<int>(ceil(RB_x)),
                                        static_cast<int>(ceil(RB_y)));

    std::vector<std::shared_ptr<LicensePlate>> lp_vector;
    lp_vector.push_back(lp);
    return lp_vector;
}

std::vector<float> DetectionMobilenet::prepareImage(const cv::Mat &frame) {
    vector<float> flattenedImage;
    int IMG_WIDTH = IMG_W, IMG_HEIGHT = IMG_H, INPUT_SIZE = IMG_C * IMG_WIDTH * IMG_HEIGHT;
    flattenedImage.resize(INPUT_SIZE, 0);
    cv::Mat resizedFrame;
    resize(frame, resizedFrame, cv::Size(IMG_WIDTH, IMG_HEIGHT));
    for (int row = 0; row < resizedFrame.rows; row++) {
        for (int col = 0; col < resizedFrame.cols; col++) {
            uchar *pixels = resizedFrame.data + resizedFrame.step[0] * row + resizedFrame.step[1] * col;
            flattenedImage[row * IMG_WIDTH + col] =
                    static_cast<float>(pixels[0] - Constants::PIXEL_MEAN_1_VALUE);

            flattenedImage[row * IMG_WIDTH + col + IMG_HEIGHT * IMG_WIDTH] =
                    static_cast<float>(pixels[1] - Constants::PIXEL_MEAN_2_VALUE);

            flattenedImage[row * IMG_WIDTH + col + 2 * IMG_HEIGHT * IMG_WIDTH] =
                    static_cast<float>(pixels[2] - Constants::PIXEL_MEAN_3_VALUE);
        }
    }
    return move(flattenedImage);
}

void DetectionMobilenet::create_anchor(vector<box> &anchor, int w, int h) {
    anchor.clear();
    std::vector<std::vector<int> > feature_map(3);
    for (int i = 0; i < feature_map.size(); ++i) {
        feature_map[i].push_back(ceil(h / steps[i]));
        feature_map[i].push_back(ceil(w / steps[i]));
    }
    for (int k = 0; k < feature_map.size(); ++k) {
        std::vector<int> min_size = min_sizes[k];
        for (int i = 0; i < feature_map[k][0]; ++i) {
            for (int j = 0; j < feature_map[k][1]; ++j) {
                for (int l = 0; l < min_size.size(); ++l) {
                    float s_kx = min_size[l] * 1.0 / w;
                    float s_ky = min_size[l] * 1.0 / h;
                    float cx = (j + 0.5) * steps[k] / w;
                    float cy = (i + 0.5) * steps[k] / h;
                    box axil = {cx, cy, s_kx, s_ky};
                    anchor.push_back(axil);
                }
            }
        }
    }
}

void DetectionMobilenet::nms(vector<bbox> &input_boxes, float NMS_THRESH) {
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float h = std::max(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH) {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            } else {
                j++;
            }
        }
    }
}

inline bool DetectionMobilenet::cmp(DetectionMobilenet::bbox a, DetectionMobilenet::bbox b) {
    if (a.s > b.s)
        return true;
    return false;
}

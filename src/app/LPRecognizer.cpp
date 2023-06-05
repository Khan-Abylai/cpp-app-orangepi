//
// Created by kartykbayev on 8/28/22.
//

#include "LPRecognizer.h"

using namespace std;
using namespace Ort;

LPRecognizer::LPRecognizer() {
    env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "ONNX_RECOGNITION");
    sessionOptions = SessionOptions();
    templateMatching = make_unique<TemplateMatching>();
    session = Ort::Session(env, Constants::recognizerModelFilepath.c_str(), sessionOptions);
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    std::vector<int64_t> inputTensorShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
    inputNames.push_back(session.GetInputName(0, allocator));
    outputNames.push_back(session.GetOutputName(0, allocator));

}

std::vector<float> LPRecognizer::prepareImages(const std::vector<cv::Mat> &frames) {

    int batchSize = frames.size();

    vector<float> flattenedImage;
    flattenedImage.resize(batchSize * INPUT_SIZE);
    cv::Mat resizedFrame;

    for (int batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        resize(frames[batchIndex], resizedFrame, cv::Size(IMG_WIDTH, IMG_HEIGHT));

        for (int row = 0; row < resizedFrame.rows; row++) {
            for (int col = 0; col < resizedFrame.cols; col++) {
                uchar *pixels = resizedFrame.data + resizedFrame.step[0] * row + resizedFrame.step[1] * col;
                flattenedImage[batchIndex * 3 * IMG_HEIGHT * IMG_WIDTH + row * IMG_WIDTH + col] =
                        static_cast<float>(pixels[0]) / Constants::PIXEL_MAX_VALUE;

                flattenedImage[batchIndex * 3 * IMG_HEIGHT * IMG_WIDTH + row * IMG_WIDTH + col +
                               IMG_HEIGHT * IMG_WIDTH] =
                        static_cast<float>(pixels[1]) / Constants::PIXEL_MAX_VALUE;

                flattenedImage[batchIndex * 3 * IMG_HEIGHT * IMG_WIDTH + row * IMG_WIDTH + col +
                               2 * IMG_HEIGHT * IMG_WIDTH] =
                        static_cast<float>(pixels[2]) / Constants::PIXEL_MAX_VALUE;
            }
        }
    }

    return move(flattenedImage);
}

std::vector<float> LPRecognizer::softmax(vector<float> &score_vec) {
    vector<float> softmax_vec(ALPHABET_SIZE);
    double score_max = *(max_element(score_vec.begin(), score_vec.end()));
    double e_sum = 0;
    for (int j = 0; j < ALPHABET_SIZE; j++) {
        softmax_vec[j] = exp((double) score_vec[j] - score_max);
        e_sum += softmax_vec[j];
    }
    for (int k = 0; k < ALPHABET_SIZE; k++)
        softmax_vec[k] /= e_sum;
    return softmax_vec;
}

std::vector<std::pair<std::string, float>> LPRecognizer::predict(const vector<cv::Mat> &frames) {

    auto flattenedImages = prepareImages(frames);
    int batchSize = frames.size();

    std::vector<int64_t> inputTensorShape{1, 3, 32, 128};
    size_t inputTensorSize = Utils::vectorProduct(inputTensorShape);
    std::vector<float> inputTensorValues(inputTensorSize);
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    std::vector<Ort::Value> inputTensors;

    vector<float> predictions;
    predictions.resize(OUTPUT_SIZE * batchSize);

    for (int batchIndex = 0; batchIndex < batchSize; batchIndex++) {

        vector<float> preds(OUTPUT_SIZE);
        std::copy(flattenedImages.begin() + batchIndex * inputTensorSize,
                  flattenedImages.begin() + inputTensorSize * (batchIndex + 1),
                  inputTensorValues.begin());

        inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorSize,
                                                               inputTensorShape.data(), inputTensorShape.size()
        ));

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

        // apply softmax
        for (int i = 0; i < SEQUENCE_SIZE; i++) {
            vector<float> temp_out(ALPHABET_SIZE);
            for (int j = 0; j < ALPHABET_SIZE; j++) {
                temp_out[j] = output[ALPHABET_SIZE * i + j];
            }

            vector<float> result = softmax(temp_out);
            temp_out.clear();
            int count = 0;
            for (int j = i * ALPHABET_SIZE; j < ALPHABET_SIZE * (i + 1); j++) {
                preds[j] = result[count];
                count++;
            }
            result.clear();
        }
        output.clear();
        int c = 0;
        for (int i = batchIndex * SEQUENCE_SIZE * ALPHABET_SIZE;
             i < (batchIndex + 1) * ALPHABET_SIZE * SEQUENCE_SIZE; i++) {
            predictions[i] = preds[c];
            c++;
        }
        preds.clear();
    }

    vector<pair<string, float>> labels;
    for (int batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float prob = 1.0;
        string currentLabel;
        currentLabel.reserve(MAX_PLATE_SIZE);

        int currentChar = BLANK_INDEX;
        float currentProb = 1.0;

        for (int i = 0; i < SEQUENCE_SIZE; i++) {
            float maxProb = 0.0;
            int maxIndex = 0;

            for (int j = 0; j < ALPHABET_SIZE; j++) {
                if (maxProb < predictions[batchIndex * ALPHABET_SIZE * SEQUENCE_SIZE + i * ALPHABET_SIZE + j]) {
                    maxIndex = j;
                    maxProb = predictions[batchIndex * ALPHABET_SIZE * SEQUENCE_SIZE + i * ALPHABET_SIZE + j];
                }
            }

            if (maxIndex == currentChar) {
                currentProb = max(maxProb, currentProb);
            } else {
                if (currentChar != BLANK_INDEX) {
                    currentLabel += ALPHABET[currentChar];
                    prob *= currentProb;
                }
                currentProb = maxProb;
                currentChar = maxIndex;
            }
        }

        if (currentChar != BLANK_INDEX) {
            currentLabel += ALPHABET[currentChar];
            prob *= currentProb;
        }

        if (currentLabel.empty()) {
            currentLabel += ALPHABET[BLANK_INDEX];
            prob = 0.0;
        }

        labels.emplace_back(make_pair(currentLabel, prob));
    }

    return labels;
}


std::pair<cv::Mat, cv::Mat> LPRecognizer::divideImageIntoHalf(const cv::Mat &lpImage) {
    lpImage(cv::Rect(0, 0, Constants::SQUARE_LP_W, Constants::SQUARE_LP_H / 2)).copyTo(
            whiteImage.colRange(0, Constants::SQUARE_LP_W).rowRange(0, Constants::RECT_LP_H));
    auto firstHalf = whiteImage.clone();

    lpImage(cv::Rect(0, Constants::SQUARE_LP_H / 2, Constants::SQUARE_LP_W,
                     Constants::SQUARE_LP_H / 2)).copyTo(
            whiteImage.colRange(0, Constants::SQUARE_LP_W).rowRange(0, Constants::RECT_LP_H));
    auto secondHalf = whiteImage.clone();

    return make_pair(firstHalf, secondHalf);
}

// gesture_detection.cpp
#include <tensorflow/c/c_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <memory>

void NoOpDeallocator(void*, size_t, void*) {}

TF_Graph* LoadGraph(const char* model_path) {
    // TensorFlow C++ API를 통한 모델 로드 함수 (TF_NewGraph와 관련된 함수들 포함)
    TF_Graph* graph = TF_NewGraph();
    TF_Buffer* buffer = TF_NewBufer();

    FILE* f = fopen(model_path, "rb");
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    std::vector<char> model_path(fsize);
    fread(model_path.data(), 1, fsize, f);
    fclose(f);

    buffer->data = model_data.data();
    buffer->length = model_data.size();

    TF_ImportGraphDefOptions* graph_opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(graph, buffer, graph_opts, status);

    if (TF_GetCode(status) != TF_OK) {
        std::cerr << "Error: Unable to load graph " << TF_Message(status) << std::endl;
        TF_DeleteGraph(graph);
        graph = nullptr;
    }

    TF_DeleteImportGraphDefOptions(graph_opts);
    TF_DeleteBuffer(buffer);
    return graph;
}

void PreprocessImage(cv::Mat& img, float* output) {
    cv::resize(img, img, cv::Size(64, 64));
    img.convertTo(img, CV_32F, 1.0 / 255);
    std::memcpy(output, img.data, img.total() * sizeof(float));
}

int PredictGesture(TF_Graph* graph, TF_Session* session, float* input_data, int input_size) {
    TF_Output input_op = {TF_GraphOperationByName(graph, "serving_default_conv2d_input"), 0};
    TF_Output output_op = {TF_GraphOperationByName(graph, "StatefulPartitionedCall"), 0};

    if (!input_op.oper || !output_op.oper) {
        std::cerr << "Failed to find input or output operations in the graph." << std::endl;
        return -1;
    }

    int64_t dims[4] = {1, 64, 64, 1};
    TF_Tensor* input_tensor = TF_NewTensor(TF_FLOAT, dims, 4, input_data, input_size * sizeof(float), &NoOpDeallocator, nullptr);

    TF_Tensor* output_tensor = nullptr;
    TF_SessionRun(session, nullptr, &input_op, &input_tensor, 1, &output_op, &output_tensor, 1, nullptr, 0, nullptr, nullptr);

    auto data = static_cast<float*>(TF_TensorData(output_tensor));
    int gesture = std::distance(data, std::max_element(data, data + 3));

    TF_DeleteTensor(input_tensor);
    TF_DeleteTensor(output_tensor);
    
    return gesture;
}

int main() {
    const char* model_path = "gesture_model";  // 학습된 모델 경로
    TF_Status* status = TF_NewStatus();
    TF_Graph* graph = LoadGraph(model_path, status);
    if (!graph) {
        std::cerr << "Failed to load model graph" << std::endl;
        return -1;
    }

    TF_SessionOptions* sess_opts = TF_NewSessionOptions();
    TF_Session* session = TF_NewSession(graph, sess_opts, status);
    TF_DeleteSessionOptions(sess_opts);
    if (TF_GetCode(status) != TF_OK) {
        std::cerr << "Error creating TensorFlow session: " << TF_Message(status) << std::endl;
        return -1;
    }

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open the camera." << std::endl;
        return -1;
    }

    cv::Mat frame, gray;
    float input_data[64 * 64];  // 모델 입력 크기

    while (cap.read(frame)) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        PreprocessImage(gray, input_data);

        // 모델 추론 호출 및 결과 가져오기
        int gesture = 0;  // 인식된 제스처 (예: gesture_1)

        // 추론 결과에 따른 동작 수행
        if (gesture == 0) {
            std::cout << "Gesture 1 detected" << std::endl;
        } else if (gesture == 1) {
            std::cout << "Gesture 2 detected" << std::endl;
        } else if (gesture == 2) {
            std::cout << "Gesture 3 detected" << std::endl;
        }

        cv::imshow("Hand Tracking", frame);
        if (cv::waitKey(10) == 27) break;  // 'ESC' 키로 종료
    }

    // 자원 해제
    TF_DeleteSession(session, status);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);
    cap.release();
    cv::destroyAllWindows();
    return 0;
}

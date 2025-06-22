#ifndef _DNN_ENGINE_H_
#define _DNN_ENGINE_H_

#ifdef OPENVINO
#include <openvino/openvino.hpp>
#endif

#ifdef TENSORRT
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferRuntime.h>
#endif

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <string>
#include <fstream>

// Dnn Inference Engine
class MyDnnEngine
{
public:
	MyDnnEngine() = default;
	~MyDnnEngine() = default;

	virtual bool InitializeEngine(const std::string& model_path, const cv::Size& size, int channel, int batch) = 0;

	virtual cv::Mat Inference(const cv::Mat& input_mat) = 0;

	virtual std::string GetEngineName() { return m_engine_name; };

protected:
	std::string m_engine_name = "";
};

// OpenCV Dnn Engine
class MyOpenCV :public MyDnnEngine
{
public:
	MyOpenCV() { m_engine_name = "OpenCV"; };
	~MyOpenCV() = default;

	virtual bool InitializeEngine(const std::string& model_path, const cv::Size& size, int channel, int batch);

	virtual cv::Mat Inference(const cv::Mat& input_mat);

private:
	cv::dnn::Net m_net;
	bool m_cuda_enabled = false;
};

// OpenVINO Dnn Engine
class MyOpenVINO: public MyDnnEngine
{
public:
	MyOpenVINO() { m_engine_name = "OpenVINO"; };
	~MyOpenVINO() = default;

#ifndef OPENVINO
	// Initialize OpenVINO inference engine
	virtual bool InitializeEngine(const std::string& model_path, const cv::Size& size, int channel, int batch) { return false; };

	// Model inference
	virtual cv::Mat Inference(const cv::Mat& input_mat) { return cv::Mat(); };
#else
	// Initialize OpenVINO inference engine
	virtual bool InitializeEngine(const std::string& model_path, const cv::Size& size, int channel, int batch);

	// Model inference
	virtual cv::Mat Inference(const cv::Mat& input_mat);

private:
	// Convert cv::Mat To ov::Tensor
	static ov::Tensor ConvertMatToTensor(const cv::Mat& cv_mat);

	// Convert ov::Tensor To cv::Mat
	static cv::Mat ConvertTensorToMat(const ov::Tensor& ov_tensor);

private:
	ov::InferRequest m_infer_request{};
#endif
};

// TensorRT Dnn Engine

#ifdef TENSORRT
class Logger : public nvinfer1::ILogger
{
public:
	void log(Severity severity, const char* msg) noexcept {}
};
#endif

class MyTensorRT : public MyDnnEngine
{
public:
	MyTensorRT() { m_engine_name = "TensorRT"; };
	~MyTensorRT() = default;

#ifndef TENSORRT
	// Initialize TensorRT inference engine
	virtual bool InitializeEngine(const std::string& model_path, const cv::Size& size, int channel, int batch) { return false; };

	// Model inference
	virtual cv::Mat Inference(const cv::Mat& input_mat) { return cv::Mat(); };
#else
	// Initialize TensorRT inference engine
	virtual bool InitializeEngine(const std::string& onnx_model_path, const cv::Size& size, int channel, int batch);

	// Model inference
	virtual cv::Mat Inference(const cv::Mat& input_mat);

private:
	nvinfer1::Dims m_input_dims;
	nvinfer1::Dims m_output_dims;
	Logger m_glogger;
	std::shared_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
#endif
};

#endif // _DNN_ENGINE_H_
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

/**
 * @class DnnEngineType
 *
 * @brief 模型推理框架类型
 *
 * @li
 * - 2025-06-24: zhongliangjian, create
 */
enum class DnnEngineType
{
	OPENCV_DNN	= 0,	// opencv-dnn
	OPENVINO	= 1,	// openvino
	TENSORRT	= 2,	// tensorrt
	NCNN		= 3		// ncnn
};

/**
 * @class ModelParams
 *
 * @brief 模型初始化参数
 *
 * @li
 * - 2025-06-24: zhongliangjian, create
 */
struct ModelParams
{
	std::string model_file;	// 模型文件路径
	cv::Size input_size;	// 输入图像尺寸
	uint32_t channel_num;	// 输入通道数量
	uint32_t batch_size;	// 批处理大小
	DnnEngineType engine;	// 推理引擎
};


// Dnn Inference Engine
class MyDnnEngine
{
public:
	MyDnnEngine() = default;
	~MyDnnEngine() = default;

	virtual bool InitializeEngine(const std::string& model_path, const cv::Size& size, int channel, int batch) = 0;

	virtual void Inference(const cv::Mat& input_blob, std::vector<cv::Mat>& output_blobs) = 0;

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

	virtual void Inference(const cv::Mat& input_blob, std::vector<cv::Mat>& output_blobs);

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
	virtual void Inference(const cv::Mat& input_blob, std::vector<cv::Mat>& output_blobs) {};
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
	virtual void Inference(const cv::Mat& input_blob, std::vector<cv::Mat>& output_blobs) {};
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

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
 * - 2023-06-24: zhongliangjian, create
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
 * - 2023-06-24: zhongliangjian, create
 */
struct ModelParams
{
	std::string model_file;	// 模型文件路径
	cv::Size input_size;	// 输入图像尺寸
	uint32_t channel_num;	// 输入通道数量
	uint32_t batch_size;	// 批处理大小
	DnnEngineType engine;	// 推理引擎
};

/**
 * @class MyDnnEngine
 *
 * @brief 通用模型推理引擎接口
 * 
 * @note 接口类，内含未实现的纯虚函数，不可直接实例化
 *
 * @li
 * - 2023-06-24: zhongliangjian, create
 */
class MyDnnEngine
{
public:
	/**
	 * @brief 构造函数
	 */
	MyDnnEngine() = default;

	/**
	 * @brief 析构函数
	 */
	~MyDnnEngine() = default;

	/**
	 * @brief  推理引擎初始化
	 * 
	 * @param  model_path  模型路径
	 * @param  size		   模型输入尺寸
	 * @param  channel     输入通道
	 * @param  batch	   批大小
	 * 
	 * @return  true/false
	 * 
	 * @li
	 * - 2023-06-24: zhongliangjian, create
	 */
	virtual bool InitializeEngine(const std::string& model_path, const cv::Size& size, int channel, int batch) = 0;

	/**
	 * @brief  执行推理
	 * 
	 * @param  input_blob  推理输入（B,C,H,W）
	 * @param  output_blobs  推理输出
	 * 
	 * @li
	 * - 2023-06-24: zhongliangjian, create
	 */
	virtual void Inference(const cv::Mat& input_blob, std::vector<cv::Mat>& output_blobs) = 0;

	/**
	 * @brief  获取推理引擎名称
	 * 
	 * @return  推理引擎名称
	 * 
	 * @li
	 * - 2023-06-24: zhongliangjian, create
	 */
	virtual std::string GetEngineName() 
	{
		return m_engine_name; 
	};

	/**
	 * @brief  创建引擎
	 * 
	 * @return  推理引擎
	 * 
	 * @li
	 * - 2023-06-24: zhongliangjian, create
	 */
	static std::unique_ptr<MyDnnEngine> CreateEngine(const ModelParams& model_params);

protected:
	std::string m_engine_name = "";
	
};

/**
 * @class MyOpenCV
 *
 * @brief OpenCV-DNN推理引擎
 *
 * @li
 * - 2023-06-24: zhongliangjian, create
 */
class MyOpenCV: public MyDnnEngine
{
public:
	/**
	 * @brief 构造函数
	 */
	MyOpenCV(): m_engine_name("OpenCV-DNN") {};

	/**
	 * @brief 构造函数
	 */
	~MyOpenCV() = default;

	/**
	 * @brief 推理引擎初始化（重写）
	 */
	virtual bool InitializeEngine(const std::string& model_path, const cv::Size& size, int channel, int batch);

	/**
	 * @brief 执行推理（重写）
	 */
	virtual void Inference(const cv::Mat& input_blob, std::vector<cv::Mat>& output_blobs);

private:
	cv::dnn::Net m_net;
	bool m_cuda_enabled = false;
};

/**
 * @class MyOpenVINO
 *
 * @brief OpenVINO推理引擎
 *
 * @li
 * - 2023-06-24: zhongliangjian, create
 */
class MyOpenVINO: public MyDnnEngine
{
public:
	/**
	 * @brief 构造函数
	 */
	MyOpenVINO(): m_engine_name("OpenVINO") {};

	/**
	 * @brief 构造函数
	 */
	~MyOpenVINO() = default;

#ifndef OPENVINO
	/**
	 * @brief 推理引擎初始化（重写）
	 */
	virtual bool InitializeEngine(const std::string& model_path, const cv::Size& size, int channel, int batch) { return false; };

	/**
	 * @brief 执行推理（重写）
	 */
	virtual void Inference(const cv::Mat& input_blob, std::vector<cv::Mat>& output_blobs) {};
#else
	/**
	 * @brief 推理引擎初始化（重写）
	 */
	virtual bool InitializeEngine(const std::string& model_path, const cv::Size& size, int channel, int batch);

	/**
	 * @brief 执行推理（重写）
	 */
	virtual void Inference(const cv::Mat& input_blob, std::vector<cv::Mat>& output_blobs);

private:
	/**
	 * @brief 转换cv::Mat到ov::Tensor
	 * 
	 * @param cv_mat 输入数据（B,C,H,W）
	 * 
	 * @return 转换结果
	 */
	static ov::Tensor ConvertMatToTensor(const cv::Mat& cv_mat);

	/**
	 * @brief 转换ov::Tensor到cv::Mat
	 * 
	 * @param ov_tensor 输入数据
	 * 
	 * @return 转换结果
	 */
	static cv::Mat ConvertTensorToMat(const ov::Tensor& ov_tensor);

private:
	ov::InferRequest m_infer_request{};
#endif
};


/**
 * @class Logger
 *
 * @brief TensorRT推理引擎日志记录器
 *
 * @li
 * - 2023-06-24: zhongliangjian, create
 */
#ifdef TENSORRT
class Logger : public nvinfer1::ILogger
{
public:
	void log(Severity severity, const char* msg) noexcept {}
};
#endif

/**
 * @class MyTensorRT
 *
 * @brief TensorRT推理引擎
 *
 * @li
 * - 2023-06-24: zhongliangjian, create
 */
class MyTensorRT : public MyDnnEngine
{
public:
	/**
	 * @brief 构造函数
	 */
	MyTensorRT(): m_engine_name("TensorRT") {};

	/**
	 * @brief 构造函数
	 */
	~MyTensorRT() = default;

#ifndef TENSORRT
	/**
	 * @brief 推理引擎初始化（重写）
	 */
	virtual bool InitializeEngine(const std::string& model_path, const cv::Size& size, int channel, int batch) { return false; };

	/**
	 * @brief 执行推理（重写）
	 */
	virtual void Inference(const cv::Mat& input_blob, std::vector<cv::Mat>& output_blobs) {};
#else
	/**
	 * @brief 推理引擎初始化（重写）
	 */
	virtual bool InitializeEngine(const std::string& model_path, const cv::Size& size, int channel, int batch);

	/**
	 * @brief 执行推理（重写）
	 */
	virtual void Inference(const cv::Mat& input_blob, std::vector<cv::Mat>& output_blobs);

private:
	nvinfer1::Dims m_input_dims;
	nvinfer1::Dims m_output_dims;
	Logger m_glogger;
	std::shared_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
#endif
};

#endif // _DNN_ENGINE_H_

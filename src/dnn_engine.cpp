#include "dnn_engine.h"
#include "com_tool.h"
#include "logger.h"

// Initialize OpenCV inference engine
bool MyOpenCV::InitializeEngine(const std::string& model_path, const cv::Size& size, int channel, int batch)
{
	if (!IsFileExists(model_path))
	{
		return false;
	}

	m_net = cv::dnn::readNetFromONNX(model_path);

	m_net.enableWinograd(false);

	if (m_cuda_enabled)
	{
		m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	}
	else
	{
		m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
		m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}

	return true;
}

// OpenCV do inference
void MyOpenCV::Inference(const cv::Mat& input_blob, std::vector<cv::Mat>& output_blobs)
{
	m_net.setInput(input_blob);
	return m_net.forward(output_blobs, m_net.getUnconnectedOutLayersNames());
}

#ifdef OPENVINO

// Initialize OpenVINO inference engine
bool MyOpenVINO::InitializeEngine(const std::string& model_path, const cv::Size& size, int channel, int batch)
{
	// Step 0. Check model file 
	if (!IsFileExists(model_path))
	{
		return false;
	}

	// Step 1. Initialize OpenVINO Runtime Core
	ov::Core core;

	// Step 2. Get list of available devices
	std::vector<std::string> available_devices = core.get_available_devices();

	if (available_devices.empty())
	{
		return false;
	}

	std::string device_name = "";
	for (auto device : available_devices)
	{
		if ("GPU" == device)
		{
			device_name = device;
			break;
		}
		
		if ("CPU" == device)
		{
			device_name = device;
		}
	}

	if ("" == device_name)
	{
		LOG_INFO("OpenVINO Running Device: None");
		return false;
	}

	LOG_INFO("OpenVINO Running Device: " + device_name);

	// Step 3. Read a model
	std::string utf8str = localToUtf8(model_path);
	std::shared_ptr<ov::Model> model = core.read_model(utf8str);
	model->reshape({ batch, channel, size.height, size.width });

	// Step 4. Configure preprocessing
	ov::preprocess::PrePostProcessor ppp(model);
	ppp.input().model().set_layout("NCHW");
	ppp.output().tensor().set_element_type(ov::element::f32);
	model = ppp.build();

	// Step 5. Loading a model to the deivce
	ov::CompiledModel compiled_model = core.compile_model(model, device_name, ov::cache_dir("dnn_cache/"));

	// Step 6. Create an infer request
	m_infer_request = compiled_model.create_infer_request();

	return true;
}

// Convert cv::Mat to ov::Tensor
ov::Tensor MyOpenVINO::ConvertMatToTensor(const cv::Mat& input_blob)
{
	LOG_DEBUG("Runing ");
	// Get input type
	ov::element::Type input_type;
	switch (input_blob.depth())
	{
		case CV_8U:
		{
			input_type = ov::element::u8;
			break;
		}
		case CV_32F:
		{
			input_type = ov::element::f32;
			break;
		}
		default:
		{
			throw("input_type exception");
			break;
		}
	}

	// Get input shape
	size_t N = input_blob.size[0];
	size_t C = input_blob.size[1];
	size_t H = input_blob.size[2];
	size_t W = input_blob.size[3];
	ov::Shape input_shape = { N, C, H, W };

	// Convert
	ov::Tensor tensor(input_type, input_shape);
	char* ptr = (char*)tensor.data();
	memcpy(ptr, (char*)input_blob.data, input_blob.total() * input_blob.elemSize());

	return tensor;
}

// Convert ov::Tensor To cv::Mat
cv::Mat MyOpenVINO::ConvertTensorToMat(const ov::Tensor& tensor)
{
	LOG_DEBUG("Runing ");
	ov::element::Type input_type = tensor.get_element_type();
	ov::Shape input_shape = tensor.get_shape();
	std::vector<int> mat_size;
	for (int axes : input_shape)
	{
		mat_size.push_back(axes);
	}
	int mat_type = 0;

	switch (input_type)
	{
		case ov::element::u8:
		{
			mat_type = CV_8U;
			break;
		}
		case ov::element::f32:
		{
			mat_type = CV_32F;
			break;
		}
		default:
		{
			throw("mat_type exception");
			break;
		}
	}

	cv::Mat mat = cv::Mat(mat_size, mat_type);

	memcpy((char*)mat.data, (char*)tensor.data(), tensor.get_byte_size());

	return mat;
}

// Openvino do inference
cv::Mat MyOpenVINO::Inference(const cv::Mat& input_mat)
{
	LOG_DEBUG("Runing ");
	// Step 1. Set up input
	ov::Tensor input_tensor = ConvertMatToTensor(input_mat);

	// Step 2. Prepare input
	m_infer_request.set_input_tensor(input_tensor);

	// Step 3. Do inference synchronously
	m_infer_request.infer();

	// Step 4. Process output
	const ov::Tensor& output_tensor = m_infer_request.get_output_tensor(0);
	cv::Mat output_mat = ConvertTensorToMat(output_tensor);

	return output_mat;
}

#endif

#ifdef TENSORRT

// Initialize TensorRT inference engine
bool MyTensorRT::InitializeEngine(const std::string& onnx_model_path, const cv::Size& size, int channel, int batch)
{
	// Step 0. Check model file
	std::string model_path = StrReplace(onnx_model_path, ".onnx", ".engine");
	if (!IsFileExists(model_path))
	{
		LOG_ERROR("model file is not exist.");
		return false;
	}

	// Step 1. De-serialize engine from file
	std::ifstream engine_file(model_path, std::ios::binary);
	if (engine_file.fail())
	{
		LOG_ERROR("engine file load failed.");
		return false;
	}

	engine_file.seekg(0, std::ifstream::end);
	auto fsize = engine_file.tellg();
	engine_file.seekg(0, std::ifstream::beg);

	std::vector<char> engine_data(fsize);
	engine_file.read(engine_data.data(), fsize);

	m_glogger = Logger();
	std::shared_ptr<nvinfer1::IRuntime> runtime{ nvinfer1::createInferRuntime(m_glogger) };
	m_engine.reset(runtime->deserializeCudaEngine(engine_data.data(), fsize, nullptr));

	if (nullptr == m_engine.get())
	{
		LOG_ERROR("engine deserialize failed.");
		return false;
	}

	m_input_dims = nvinfer1::Dims4{ batch, channel, size.height, size.width };

	return true;
}

// Model inference
cv::Mat MyTensorRT::Inference(const cv::Mat& input_mat)
{
	auto context = std::shared_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
	if (!context)
	{
		LOG_ERROR("execute context create failed.");
		return cv::Mat();
	}

	auto input_idx = m_engine->getBindingIndex("input");
	if (-1 == input_idx)
	{
		LOG_ERROR("get input binding index failed.");
		return cv::Mat();
	}
	assert(m_engine->getBindingDataType(input_idx) == nvinfer1::DataType::kFLOAT);
	auto input_dims = nvinfer1::Dims4{ input_mat.size[0], input_mat.size[1], input_mat.size[2], input_mat.size[3] };
	context->setBindingDimensions(input_idx, input_dims);
	size_t input_size = std::accumulate(input_dims.d, input_dims.d + input_dims.nbDims, 1, std::multiplies<int64_t>()) * sizeof(float);

	auto output_idx = m_engine->getBindingIndex("output");
	if (-1 == output_idx)
	{
		LOG_ERROR("get output binding index failed.");
		return cv::Mat();
	}
	assert(m_engine->getBindingDataType(output_idx) == nvinfer1::DataType::kFLOAT);
	auto output_dims = context->getBindingDimensions(output_idx);
	auto output_size = std::accumulate(input_dims.d, input_dims.d + input_dims.nbDims, 1, std::multiplies<int64_t>()) * sizeof(float);

	// Allocate CUDA memory for input and output bindings
	void* input_memory = nullptr;
	if (cudaMalloc(&input_memory, input_size) != cudaSuccess)
	{
		LOG_ERROR("input cuda memory allocation failed, size = " + int2str(input_size) + "bytes");
		return cv::Mat();
	}
	void* output_memory = nullptr;
	if (cudaMalloc(&output_memory, output_size) != cudaSuccess)
	{
		LOG_ERROR("output cuda memory allocation failed, size = " + int2str(output_size) + "bytes");
		return cv::Mat();
	}

	// Create CUDA stream
	cudaStream_t stream;
	if (cudaStreamCreate(&stream) != cudaSuccess)
	{
		LOG_ERROR("cuda stream creation failed.");
		return cv::Mat();
	}

	// Copy input data to input binding memory
	auto input_buffer = std::unique_ptr<float>{ new float[input_size / sizeof(float)] };
	memcpy(input_buffer.get(), input_mat.ptr<float>(0), input_size);
	if (cudaMemcpyAsync(input_memory, input_buffer.get(), input_size, cudaMemcpyHostToDevice, stream) != cudaSuccess)
	{
		LOG_ERROR("cuda memory copy of input failed, size = " + int2str(input_size) + "bytes");
		return cv::Mat();
	}

	// Run TensorRT inference
	void* bindings[] = { input_memory, output_memory };
	if (false == context->enqueueV2(bindings, stream, nullptr))
	{
		LOG_ERROR("TessorRT inference failed.");
		return cv::Mat();
	}

	// Copy output data to output binding memory
	auto output_buffer = std::unique_ptr<float>{ new float[output_size] };
	if (cudaMemcpyAsync(output_buffer.get(), output_memory, output_size, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
	{
		LOG_ERROR("cuda memory copy of output failed, size = " + int2str(output_size) + "bytes");
		return cv::Mat();
	}
	cudaStreamSynchronize(stream);

	// Convert tensor to cv::mat
	std::vector<int> shape(output_dims.nbDims);
	for (int i = 0; i < output_dims.nbDims; ++i)
	{
		shape[i] = output_dims.d[i];
	}
	cv::Mat output_mat = cv::Mat(shape.size(), shape.data(), CV_32F, output_buffer.get()).clone();

	// Free CUDA resources
	cudaFree(input_memory);
	cudaFree(output_memory);

	return output_mat;
}

#endif

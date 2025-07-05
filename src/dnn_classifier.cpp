#include "dnn_classifier.h"

/**
 * @brief 构造函数
 */
DnnClassifier::DnnClassifier(const ModelParams& model_params, const ClassifyParams& classify_params)
{
	// 参数初始化
	m_model_params = model_params
	m_classify_params = classify_params;

	// 获取引擎
	m_model = MyDnnEngine::CreateEngine(model_params);
}

/**
 * @brief 获取图像分类结果
 *
 * @param input 输入图像
 * @param outputs 分类结果
 *
 * @return
 *
 * @li
 * - 2025-06-24: zhongliangjian, create
 */
void DnnClassifier::GetClassifyResult(const cv::Mat& input, std::vector<ClassifyInfo>& outputs)
{

}

/**
 * @brief 前处理
 *
 * @param input 输入图像
 * @param output 输出图像（预处理后的图像）
 * 
 * @return 
 *
 * @li
 * - 2025-06-24: zhongliangjian, create
 */
void DnnClassifier::preprocess(const cv::Mat& input, cv::Mat& output)
{

}

/**
 * @brief 模型推理
 *
 * @param input 输入
 * @param output 输出
 *
 * @return
 *
 * @li
 * - 2025-06-24: zhongliangjian, create
 */
void DnnClassifier::inference(const cv::Mat& input, std::vector<cv::Mat>& outputs)
{

}

/**
 * @brief 后处理
 *
 * @param inputs 输入图像
 * @param outputs 输出结果
 *
 * @return
 *
 * @li
 * - 2025-06-24: zhongliangjian, create
 */
void DnnClassifier::postprocess(const std::vector<cv::Mat>& inputs, std::vector<ClassifyInfo>& outputs)
{

}
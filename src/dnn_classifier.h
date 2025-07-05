/**
 * @file dnn_classifier.h
 *
 * @brief 图像分类器
 *
 * @author zhongliangjian
 *
 * @date 2025-06-24
 */

#ifndef _DNN_CLASSIFIER_H_
#define _DNN_CLASSIFIER_H_

#include <memory>
#include "dnn_engine.h"

/**
* @class ClassifyParams
*
* @brief 分类器参数
*
* @li
* - 2025-06-24: zhongliangjian, create
*/
struct ClassifyParams
{
	float score_thresh{ 0.45 };
	int class_num{ 1 };
};

/**
* @class ClassifyInfo
*
* @brief 分类信息
*
* @li
* - 2025-06-24: zhongliangjian, create
*/
struct ClassifyInfo
{
	int class_id;
	float class_score;
};

/**
* @class DnnClassifier
*
* @brief 图像分类器
*
* @li
* - 2025-06-24: zhongliangjian, create
*/
class DnnClassifier
{
public:
	/**
	 * @brief 构造函数
	 */
	DnnClassifier(const ModelParams& model_params, const ClassifyParams& classify_params);

	/**
	 * @brief 析构函数
	 */
	~DnnClassifier() = default;

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
	void GetClassifyResult(const cv::Mat& input, std::vector<ClassifyInfo>& outputs);

private:
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
	void preprocess(const cv::Mat& input, cv::Mat& output);

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
	void inference(const cv::Mat& input, std::vector<cv::Mat>& outputs);

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
	void postprocess(const std::vector<cv::Mat>& inputs, std::vector<ClassifyInfo>& outputs);

private:
	std::unique_ptr<MyDnnEngine> m_model;
	ModelParams m_model_params;
	ClassifyParams m_classify_params;
};
#endif // _DNN_CLASSIFIER_H_

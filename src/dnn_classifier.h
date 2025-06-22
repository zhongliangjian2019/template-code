///
///	dnn图像分类器实现
///


#ifndef _DNN_CLASSIFIER_H_
#define _DNN_CLASSIFIER_H_

#include <memory>
#include "com_tool.h"
#include "logger.h"
#include "cv_tool.h"

// 微滴分类器
class DropletClassifier : public Network
{
public:
	// 构造函数
	DropletClassifier(const std::string& model_file, InferEngine frame, const cv::Size& size = cv::Size(65, 65), int in_channel = 1, int num_classes = 2, int batch = 128);

	// 析构函数
	~DropletClassifier() {};

	// 获取微滴分类结果
	void GetClassifyResult(const cv::Mat& src, const std::vector<cv::Point>& center_vec, const std::vector<int>& radius_vec, std::vector<DropletType>& categroy_vec, float score_thresh);

private:
	// 模型预测
	void Predict(const cv::Mat& src, const std::vector<cv::Point>& center_vec, const std::vector<int>& radius_vec, const std::vector<DropletType>& categroy_vec,
		std::vector<size_t>& idx_vec, std::vector<int>& cla_vec, std::vector<float>& cof_vec);

	// 模型预测前处理
	void Preprocess(const cv::Mat& src, const std::vector<cv::Point>& center_vec, const std::vector<int>& radius_vec, const std::vector<DropletType>& categroy_vec,
		std::vector<size_t>& idx_vec, std::vector<cv::Mat>& image_vec);

	// 模型预测后处理
	void Postprocess(const cv::Mat& src, std::vector<int>& ids, std::vector<float>& scores);
};
#endif // _DNN_CLASSIFIER_H_

#include "dnn_classifier.h"
#include "cv_tool.h"

///
/// @brief  分类网络构造函数
///     
/// @param[in]  model_file  
/// @param[in]  size  
/// @param[in]  in_channel  
/// @param[in]  num_classes  
/// @param[in]  batch  
///
/// @return  
///
/// @par History:
/// @li 6883/ZhongLiangJian, 2023/2/9-11:09:17
///
DropletClassifier::DropletClassifier(const std::string& model_file, InferEngine frame, const cv::Size& size, int in_channel, int num_classes, int batch) :
    Network(model_file, frame, size, in_channel, num_classes, batch) {}


///
/// @brief  获取微滴分类结果
///     
/// @param[in]  src  输入图像（微滴图）
/// @param[in/out]  droplet  微滴信息
///
/// @return  
///
/// @par History:
/// @li 6883/ZhongLiangJian, 2023/2/9-11:23:19
///
void DropletClassifier::GetClassifyResult(const cv::Mat& src, const std::vector<cv::Point>& center_vec, const std::vector<int>& radius_vec, std::vector<DropletType>& categroy_vec, float score_thresh)
{
	LOG_DEBUG("Running ");

	std::vector<size_t> idx_vec;
	std::vector<int> cla_vec;
	std::vector<float> cof_vec;

	Predict(src, center_vec, radius_vec, categroy_vec, idx_vec, cla_vec, cof_vec);

	// 获取微滴类别
	for (size_t i = 0; i < idx_vec.size(); ++i)
	{
		size_t index = idx_vec[i];
		if (cof_vec[i] > score_thresh)	// 只修改大于阈值的结果, 未修改的结果默认保持默认
		{
			categroy_vec[index] = DropletType(cla_vec[i]);
		}
	}
}

///
/// @brief  模型预测
///     
/// @param[in]  src  输入图像
/// @param[in]  ids  
/// @param[in]  scores  
///
/// @return  
///
/// @par History:
/// @li 6883/ZhongLiangJian, 2023/2/9-11:25:34
///

void DropletClassifier::Predict(const cv::Mat& src, const std::vector<cv::Point>& center_vec, const std::vector<int>& radius_vec, const std::vector<DropletType>& categroy_vec, 
	std::vector<size_t>& idx_vec, std::vector<int>& cla_vec, std::vector<float>& cof_vec)
{
	LOG_DEBUG("Running ");

	std::vector<cv::Mat> image_vec;
	Preprocess(src, center_vec, radius_vec, categroy_vec, idx_vec, image_vec);
	
	std::vector<cv::Mat> batch_images;
	std::vector<int> batch_cla;
	std::vector<float> batch_cof;
	cv::Mat batch_out;

	cla_vec.clear();
	cof_vec.clear();

	for (size_t i = 0; i < idx_vec.size(); ++i)
	{
		batch_images.push_back(image_vec[i]);

		if ((m_batch == batch_images.size()) || (i == idx_vec.size() - 1))
		{
			Inference(batch_images, batch_out, 1);
			Postprocess(batch_out, batch_cla, batch_cof);
			cla_vec.insert(cla_vec.end(), batch_cla.begin(), batch_cla.end());
			cof_vec.insert(cof_vec.end(), batch_cof.begin(), batch_cof.end());

			batch_images.clear();
			batch_cla.clear();
			batch_cof.clear();
		}
	}
}

///
/// @brief  模型预测前处理
///     
/// @param[in]  src  输入图像, 单通道灰度图
/// @param[in]  dst  处理后的图像
///
/// @return  
///
/// @par History:
/// @li 6883/ZhongLiangJian, 2023/2/9-11:26:50
///
void DropletClassifier::Preprocess(const cv::Mat& src,  const std::vector<cv::Point>& center_vec, const std::vector<int>& radius_vec, const std::vector<DropletType>& categroy_vec, 
	std::vector<size_t>& idx_vec, std::vector<cv::Mat>& image_vec)
{
	LOG_DEBUG("Running ");

	idx_vec.clear();

	// 找出可疑区域
	cv::Mat thresh;
	cv::threshold(src, thresh, 127, 255, cv::THRESH_BINARY);
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
	cv::morphologyEx(thresh, thresh, cv::MORPH_OPEN, kernel);

	cv::Mat input_image = src.clone();

	// 图像增强
	//input_image = ImageAugment::GammaTransform(input_image);

	// 输入通道检查
	if (input_image.channels() != m_channel)
	{
		if (1 == m_channel)
		{
			cv::cvtColor(input_image, input_image, cv::COLOR_BGR2GRAY);
		}
		else
		{
			cv::cvtColor(input_image, input_image, cv::COLOR_GRAY2BGR);
		}
	}

	// 图像切分
	for (size_t i = 0; i < center_vec.size(); ++i)
	{
		if (DropletType::NORMAL != categroy_vec[i])
		{
			continue;
		}
		cv::Mat mark = CVTool::GetImageRoi(thresh, center_vec[i], radius_vec[i] * 2 + 1).clone();
		if (mark.empty())
		{
			continue;
		}
		if (cv::countNonZero(mark) == 0)
		{
			continue;
		}

		cv::Mat image = CVTool::GetImageRoi(input_image, center_vec[i], radius_vec[i] * 6 + 1).clone();
		if (image.empty())
		{
			continue;
		}

		idx_vec.push_back(i);

		if (image.size() != m_size)
		{
			cv::resize(image, image, m_size);
		}
		image_vec.push_back(image.clone());
	}
}

///
/// @brief  分类结果后处理
///     
/// @param[in]  src  模型输出
/// @param[in]  ids  类别ID
/// @param[in]  scores  类别得分
///
/// @return  None
///
/// @par History:
/// @li 6883/ZhongLiangJian, 2023/2/9-11:16:06
///
void DropletClassifier::Postprocess(const cv::Mat& src, std::vector<int>& idx_vec, std::vector<float>& score_vec)
{
	LOG_DEBUG("Running ");

	cv::Mat ids, scores;
	Argmax(src, ids, scores);

	idx_vec.clear();
	score_vec.clear();
	for (int i = 0; i < ids.size[0]; ++i)
	{
		idx_vec.push_back(ids.ptr<uchar>(i)[0]);
		score_vec.push_back(scores.ptr<float>(i)[0]);
	}
}
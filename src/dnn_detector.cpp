#include "dnn_detector.h"
#include "cv_tool.h"
#include "com_tool.h"

/**
 * @brief 构造函数
 */
DnnDetector::DnnDetector(const ModelParams& model_params, const DetectParams& detect_params)
{
	// 初始化参数
	m_model_params = model_params;
	m_detect_params = detect_params;

	// 设置推理引擎
	m_model = MyDnnEngine::CreateEngine(model_params);
}

/**
 * @brief 获取目标检测结果
 *
 * @param input 输入图像
 * @param outputs 检测结果
 *
 * @return
 *
 * @li
 * - 2025-06-24: zhongliangjian, create
 */
void DnnDetector::getDetectResult(const cv::Mat& input, std::vector<DetectBbox>& outputs)
{
	// 1.前处理
	int pad_x = 0;
	int pad_y = 0;
	float scale = 0.0f;
	cv::Mat input_blob;
	preprocess(input, input_blob, pad_x, pad_y, scale);

	// 2.推理
	std::vector<cv::Mat> output_blobs;
	inference(input_blob, output_blobs);

	// 3.后处理
	postprocess(output_blobs, outputs, pad_x, pad_y, scale);
}

/**
 * @brief 获取实例分割结果
 *
 * @param input 输入图像
 * @param outputs 检测结果
 *
 * @return
 *
 * @li
 * - 2025-06-24: zhongliangjian, create
 */
void DnnDetector::getInstanceResult(const cv::Mat& input, std::vector<InstanceBbox>& outputs)
{
	// 1.前处理
	int pad_x = 0;
	int pad_y = 0;
	float scale = 0.0f;
	cv::Mat input_blob;
	preprocess(input, input_blob, pad_x, pad_y, scale);

	// 2.推理
	std::vector<cv::Mat> output_blobs;
	inference(input_blob, output_blobs);

	// 3.后处理
	postprocess(output_blobs, outputs, pad_x, pad_y, scale);
}

/**
* @brief 前处理
*
* @param input 输入图像
* @param output 输出图像（预处理后的图像）
* @param pad_x 宽度方向填充量
* @param pad_y 高度方向填充量
* @param scale 输入到模型的缩放尺度
*
* @return
*
* @li
* - 2025-06-24: zhongliangjian, create
*/
void DnnDetector::preprocess(const cv::Mat& input, cv::Mat& output, int& pad_x, int& pad_y, float& scale)
{
	// 1.填充图像到方形
	cv::Mat square = CVTool::FormatToSquare(input, false, pad_x, pad_y);

	// 2.预处理
	output = cv::dnn::blobFromImage(square, 1.0 / 255.0, m_model_params.input_size, cv::Scalar(), true);

	// 3.计算缩放尺度
	scale = square.rows / (float)m_model_params.input_size.height;
}

/**
 * @brief 模型推理
 *
 * @param input 输入
 * @param outputs 输出
 *
 * @return
 *
 * @li
 * - 2025-06-24: zhongliangjian, create
 */
void DnnDetector::inference(const cv::Mat& input, std::vector<cv::Mat>& outputs)
{
	m_model->Inference(input, outputs);
}

/**
 * @brief 目标检测后处理
 *
 * @param inputs 输入图像
 * @param outputs 输出结果
 * @param pad_x 预处理中的宽度填充量
 * @param pad_y 预处理中的高度填充量
 * @param scale 预处理中的缩放尺度
 *
 * @return
 *
 * @li
 * - 2025-06-24: zhongliangjian, create
 */
void DnnDetector::postprocess(const std::vector<cv::Mat>& inputs, std::vector<DetectBbox>& outputs, int pad_x, int pad_y, float scale)
{
	cv::Mat input = inputs.at(0);
	int batch = input.size[0];		// batch = 1
	int dims = input.size[1];		// dims = bbox(cx, cy, w, h) + class_num
	int rows = input.size[2];		// rows = 8400
	input = input.reshape(1, dims);
	cv::transpose(input, input);

	float* data_ptr = (float*)input.data;
	std::vector<int> class_ids;
	std::vector<float> scores;
	std::vector<cv::Rect> bboxes;
	// 从每个目标的维度开始解码
	for (int i = 0; i < rows; ++i)
	{
		// 获取分类id和得分
		float* classes_scores = data_ptr + 4;	// 偏移bbox占用的4个地址位置
		cv::Mat multi_class_scores(1, m_detect_params.class_num, CV_32FC1, classes_scores);
		cv::Point class_id;
		double class_score;
		cv::minMaxLoc(multi_class_scores, nullptr, &class_score, nullptr, &class_id);
		
		if (class_score > m_detect_params.score_thresh)
		{
			// 获取bbox
			float x = data_ptr[0];	// cx
			float y = data_ptr[1];	// cy
			float w = data_ptr[2];
			float h = data_ptr[3];

			int left = int((x - 0.5 * w) * scale - pad_x);
			int top = int((y - 0.5 * h) * scale - pad_y);
			int width = int(w * scale);
			int height = int(h * scale);
			
			bboxes.push_back(cv::Rect(left, top, width, height));

			// 获取类别id/score
			scores.push_back(class_score);
			class_ids.push_back(class_id.x);
		}

		// 偏移数据指针到下一位置
		data_ptr += dims;
	}

	// nms后处理
	std::vector<int> nms_idxes;
	multiClassesNMS(bboxes, class_ids, scores, nms_idxes, m_detect_params.class_num, m_detect_params.score_thresh, m_detect_params.nms_thresh);

	// 组织输出
	for (auto idx: nms_idxes)
	{
		DetectBbox bbox;
		bbox.bbox = bboxes.at(idx);
		bbox.class_id = class_ids.at(idx);
		bbox.class_score = scores.at(idx);
		outputs.push_back(bbox);
	}
}

/**
 * @brief 实例分割后处理
 *
 * @param inputs 输入图像
 * @param outputs 输出结果
 *
 * @return
 *
 * @li
 * - 2025-06-24: zhongliangjian, create
 */
void DnnDetector::postprocess(const std::vector<cv::Mat>& inputs, std::vector<InstanceBbox>& outputs, int pad_x, int pad_y, float scale)
{
	// 目标检测分支预测转换
	cv::Mat bbox_input = inputs.at(0);
	int batch = bbox_input.size[0];		// batch = 1
	int dims = bbox_input.size[1];		// dims = bbox(cx, cy, w, h) + class_num
	int rows = bbox_input.size[2];		// rows = 8400
	bbox_input = bbox_input.reshape(1, dims);
	cv::transpose(bbox_input, bbox_input);

	float* data_ptr = (float*)bbox_input.data;
	std::vector<int> class_ids;
	std::vector<float> scores;
	std::vector<cv::Rect> bboxes;
	std::vector<cv::Mat> features;
	// 从每个目标的维度开始解码
	for (int i = 0; i < rows; ++i)
	{
		// 获取分类id和得分
		float* curr_ptr = data_ptr + 4;	// 偏移4个位置到分类得分起始位置
		cv::Mat multi_class_scores(1, m_detect_params.class_num, CV_32FC1, curr_ptr);
		cv::Point class_id;
		double class_score;
		cv::minMaxLoc(multi_class_scores, nullptr, &class_score, nullptr, &class_id);

		if (class_score > m_detect_params.score_thresh)
		{
			// 获取bbox
			float x = data_ptr[0];	// cx
			float y = data_ptr[1];	// cy
			float w = data_ptr[2];
			float h = data_ptr[3];

			int left = int((x - 0.5 * w) * scale - pad_x);
			int top = int((y - 0.5 * h) * scale - pad_y);
			int width = int(w * scale);
			int height = int(h * scale);

			bboxes.push_back(cv::Rect(left, top, width, height));

			// 获取类别id/score
			scores.push_back(class_score);
			class_ids.push_back(class_id.x);

			// 提取mask特征
			curr_ptr += m_detect_params.class_num;	// 偏移类别个数位置到掩模特征起始位置
			cv::Mat feature(1, dims - 4 - m_detect_params.class_num, CV_32FC1, curr_ptr);
			features.push_back(feature.clone());
		}

		// 偏移数据指针到下一位置
		data_ptr += dims;
	}

	// nms后处理
	std::vector<int> nms_indexes;
	multiClassesNMS(bboxes, class_ids, scores, nms_indexes, m_detect_params.class_num, m_detect_params.score_thresh, m_detect_params.nms_thresh);

	// 提取nms结果
	std::vector<int> nms_class_ids;
	std::vector<float> nms_scores;
	std::vector<cv::Rect> nms_bboxes;
	std::vector<cv::Mat> nms_features;
	for (auto idx : nms_indexes)
	{
		nms_class_ids.push_back(class_ids.at(idx));
		nms_scores.push_back(scores.at(idx));
		nms_bboxes.push_back(bboxes.at(idx));
		nms_features.push_back(features.at(idx));
	}

	// mask解码处理
	// 掩模分支预测转换
	cv::Mat mask_input = inputs.at(1);
	int mask_dims = mask_input.size[1];
	int mask_height = mask_input.size[2];
	int mask_width = mask_input.size[3];
	mask_input = mask_input.reshape(1, {mask_dims, mask_height * mask_width});
	cv::Mat mask_feat;
	cv::vconcat(nms_features, mask_feat);
	cv::Mat masks = mask_feat * mask_input;
	cv::transpose(masks, masks);
	masks = masks.reshape(masks.cols, { mask_height, mask_width });

	
	// 提取实例
	float mask_scale = scale * m_model_params.input_size.width / mask_width;
	int gap = 1;
	for (size_t i = 0; i < nms_bboxes.size(); ++i)
	{
		// 缩放bbox到mask尺度
		cv::Rect bbox = nms_bboxes.at(i);
		bbox.x = std::max(0, int(bbox.x / mask_scale) - gap);
		bbox.y = std::max(0, int(bbox.y / mask_scale) - gap);
		bbox.width = int(bbox.width / mask_scale) + 2 * gap;
		bbox.height = int(bbox.height / mask_scale) + 2 * gap;

		// 提取掩模
		cv::Mat mask;
		cv::extractChannel(masks, mask, i);

		// 扣取roi区域
		mask = mask(bbox).clone();
		cv::resize(mask, mask, cv::Size(), mask_scale, mask_scale);
		cv::threshold(mask, mask, 5, 255, cv::THRESH_BINARY);

		// 提取轮廓
		mask.convertTo(mask, CV_8UC1);
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		// 提取最大轮廓
		std::vector<cv::Point> contour;
		int max_idx = CVTool::GetMaxContour(contours);
		if (max_idx != -1)
		{
			// 面积阈值
			double area = cv::contourArea(contours.at(max_idx));
			if (area > (mask.rows * mask.cols * 0.25))
			{
				// 轮廓近似
				double epsilon = 0.003 * cv::arcLength(contours.at(max_idx), true);	// 参数0.003为经验值
				cv::approxPolyDP(contours.at(max_idx), contour, epsilon, true);

				// 基准位置偏移
				for (auto& point : contour)
				{
					point.x += int(bbox.x * mask_scale);
					point.y += int(bbox.y * mask_scale);
				}

			}
		}
		
		// 组合结果
		DetectBbox det_bbox;
		det_bbox.bbox = nms_bboxes.at(i);
		det_bbox.class_id = nms_class_ids.at(i);
		det_bbox.class_score = nms_scores.at(i);

		InstanceBbox ins_bbox;
		ins_bbox.det_bbox = det_bbox;
		ins_bbox.ins_contour = contour;

		outputs.push_back(ins_bbox);
	}
}

/**
 * @brief 多类别NMS处理
 *
 * @param bboxes 边界框
 * @param classes_ids 类别id
 * @param scores 类别得分
 * @param indexes nms后的结果索引
 * @param num_classes 分类类别总数
 * @param score_thresh 得分阈值
 * @param nms_thresh iou阈值
 * 
 * @return
 *
 * @li
 * - 2025-06-24: zhongliangjian, create
 */
void DnnDetector::multiClassesNMS(const std::vector<cv::Rect>& bboxes, const std::vector<int>& class_ids, const std::vector<float>& scores,
	std::vector<int>& indexes, int num_classes, float score_thresh, float nms_thresh)
{
	// 检查数据匹配度
	if (bboxes.size() != class_ids.size() || bboxes.size() != scores.size())
	{
		LOG_WARNING("nms data size not match, bboxes.size = " + int2str(bboxes.size()) \
			+ ", class_ids.size = " + int2str(class_ids.size()) \
			+ ", scores.size = " + int2str(scores.size()));
		return;
	}

	// 按类别分组全局索引位置
	std::vector<std::vector<int>> class_indexes(num_classes, std::vector<int>());
	for (size_t i = 0; i < class_ids.size(); ++i)
	{
		int cla_id = class_ids.at(i);
		class_indexes.at(cla_id).push_back(int(i));
	}

	// 分类NMS后处理
	for (int cla = 0; cla < num_classes; ++cla)
	{
		// 当前类别的全局索引
		const auto& curr_indexes = class_indexes.at(cla);

		// 根据当前类别的全局索引提取当前类别的bbox/score
		std::vector<cv::Rect> curr_bboxes;
		curr_bboxes.reserve(curr_indexes.size());
		std::vector<float> curr_scores;
		curr_scores.reserve(curr_indexes.size());

		for (auto glob_idx : curr_indexes)
		{
			curr_bboxes.push_back(bboxes.at(glob_idx));
			curr_scores.push_back(scores.at(glob_idx));
		}

		// nms处理获取局部索引
		std::vector<int> nms_result;
		cv::dnn::NMSBoxes(curr_bboxes, curr_scores, score_thresh, nms_thresh, nms_result);

		// 根据局部id提取nms后的全局id
		for (auto loc_idx : nms_result)
		{
			indexes.push_back(curr_indexes.at(loc_idx));
		}
	}
}
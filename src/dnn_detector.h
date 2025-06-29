/**
 * @file dnn_detector.h
 *
 * @brief yolo目标检测/实例分割
 *
 * @author zhongliangjian
 *
 * @date 2025-06-24
 */
#ifndef _DNN_DETECTOR_H_
#define _DNN_DETECTOR_H_

#include <string>
#include <memory>

#include "dnn_engine.h"
#include "logger.h"

 /**
 * @class DetectBbox
 *
 * @brief 目标检测边界框结构体
 *
 * @li
 * - 2025-06-24: zhongliangjian, create
 */
struct DetectBbox
{
	cv::Rect bbox;		// 边界框
	int class_id;		// 类别id
	float class_score;	// 类别得分
};

/**
* @class InstanceBbox
*
* @brief 实例分割结果结构体
*
* @li
* - 2025-06-24: zhongliangjian, create
*/
struct InstanceBbox
{
	DetectBbox det_bbox;				// 检测框
	std::vector<cv::Point> ins_contour;	// 实例轮廓
};

/**
* @class InstanceBbox
*
* @brief 实例分割结果结构体
*
* @li
* - 2025-06-24: zhongliangjian, create
*/
struct DetectParams
{
	float score_thresh{ 0.45 };
	float nms_thresh{ 0.5 };
	int class_num{ 1 };
};

/**
* @class DnnDetector
*
* @brief yolo目标检测/实例分割类
*
* @li
* - 2025-06-24: zhongliangjian, create
*/
class DnnDetector
{
public:
	/**
	 * @brief 构造函数
	 */
	DnnDetector(const ModelParams& model_params, const DetectParams& detect_params);

	/**
	 * @brief 析构函数
	 */
	~DnnDetector() = default;

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
	void getDetectResult(const cv::Mat& input, std::vector<DetectBbox>& outputs);

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
	void getInstanceResult(const cv::Mat& input, std::vector<InstanceBbox>& outputs);

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
	void preprocess(const cv::Mat& input, cv::Mat& output, int& pad_x, int& pad_y, float& scale);

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
	void postprocess(const std::vector<cv::Mat>& inputs, std::vector<DetectBbox>& outputs, int pad_x, int pad_y, float scale);

	/**
	 * @brief 实例分割后处理
	 *
	 * @param input 输入图像
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
	void postprocess(const std::vector<cv::Mat>& inputs, std::vector<InstanceBbox>& outputs, int pad_x, int pad_y, float scale);

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
	void multiClassesNMS(const std::vector<cv::Rect>& bboxes, const std::vector<int>& class_ids, const std::vector<float>& scores,
		std::vector<int>& indexes, int num_classes, float score_thresh, float nms_thresh);

private:
	std::unique_ptr<MyDnnEngine> m_model;
	ModelParams m_model_params;
	DetectParams m_detect_params;
};


#endif // _DNN_DETECTOR_H_
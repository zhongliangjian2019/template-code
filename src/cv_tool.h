///
/// 自定义OpenCV工具函数
///
#ifndef _CV_TOOL_H_
#define _CV_TOOL_H_

#include <string>
#include <opencv2/opencv.hpp>

// Opencv工具
namespace CVTool
{  
    // 获取轮廓质心(单个)
    cv::Point GetContourCenter(const std::vector<cv::Point> &contour);

    // 获取轮廓质心(多个)
    void GetContourCenter(const std::vector<std::vector<cv::Point>> &contours, std::vector<cv::Point> &centers);

    // 获取轮廓中心与半径
    void GetContourCenterAndRadius(const std::vector<cv::Point>& contour, cv::Point& center, int& radius, float scale = 0.75);

	// 获取二值图像轮廓中心
	void GetBinaryContourCenter(const cv::Mat& src, std::vector<cv::Point>& centers, int area_min = 25, int area_max = 1000);

    // 计算图像百分位数
    float ImagePercentile(cv::Mat& image, size_t percent);

    // 移除图像噪声
    cv::Mat RemoveImageNoise(const cv::Mat& input_image, int kernel_size = 5);

    // 按中心裁剪图像
    cv::Mat CenterCropImage(const cv::Mat& src_image, const cv::Size& dsize = cv::Size(-1, -1));

    // 图像分块
    void GetSplitBlockImages(const cv::Mat& srcImage, const cv::Size& blockSize, std::vector<std::vector<cv::Mat>>& blockImages);

    // 根据中心点获取图像感兴趣区域
    cv::Mat GetImageRoi(const cv::Mat& src, const cv::Point& center, int size);

	// 图像平移（单张）
    cv::Mat MoveImage(const cv::Mat& srcImage, const cv::Point& offset);

    // 图像平移（多张）
    void MoveImages(const std::vector<cv::Mat>& srcImages, const std::vector<cv::Point>& offsets, std::vector<cv::Mat>& dstImages);

    // 图像分块
    void SplitImageBlock(const cv::Mat& src, std::vector<cv::Mat>& block_images, std::vector<cv::Point>& block_indexs,
        cv::Size& pad_image_size, const cv::Size& block_size = cv::Size(512, 512), float overlap_ratio = 0.15);

    void SplitImageBlockA(const cv::Mat& src, std::vector<cv::Mat>& block_images, std::vector<cv::Point>& block_offsets,
        const cv::Size& block_size, float overlap_ratio);

    // 图像拼接
	cv::Mat MosaicImageBlock(const std::vector<cv::Mat>& block_images, const std::vector<cv::Point>& block_indexs, const cv::Size& pad_size, const cv::Size& src_size, float overlap_ratio = 0.15);

	// 形态学膨胀重建
	cv::Mat MorphRestructure(const cv::Mat& temp, const cv::Mat& mark, int connect = 4);

	// 形态学击中与击不中变换
	cv::Mat MorphHitAndMissTransform(const cv::Mat& src, const cv::Mat& kernel_1, const cv::Mat& kernel_2);

	// 形态学孔洞填充
	cv::Mat MorphFillHole(const cv::Mat& src, int connect = 4);

    // 形态学边界对象移除
    cv::Mat MorphRemoveEdgeObject(const cv::Mat& src, int connect = 4);

    void AutoCropImages(const std::vector<cv::Mat>& src_vec, std::vector<cv::Mat>& dst_vec, const cv::Size& dst_size = cv::Size());

    // 自动图像裁剪
    void AutoCropImages(const std::vector<std::string>& image_paths, const cv::Size& result_size = cv::Size(1641, 1641));

    // 读取图像
    cv::Mat ReadImage(const std::string& image_path, int flag = 0);

    // 保存图像
    void SaveImage(const cv::Mat& image, const std::string& save_path);

    ///
	/// @brief  对数变换：拉升亮度和对比度
	///     
	/// @param[in]  input  输入图像
	/// @param[in]  light_min  亮度低值（确定背景亮度变化范围）
	/// @param[in]  light_max  亮度高值（确定前景亮度变化范围）
	///
	/// @return  增强后的图像
	///
	/// @par History:
	/// @li 6883/ZhongLiangJian, 2022/12/7-17:13:31
	///
	cv::Mat LogTransform(const cv::Mat& input, int light_min = 50, int light_max = 200, float drop_ratio = 0.25);

	///
	/// @brief  反锐化掩膜：增强高频信息
	///     
	/// @param[in]  input  输入图像
	/// @param[in]  ksize  高斯模糊核的大小
	/// @param[in]  scale  增益因子（值在0-1之间，值越大增益越大）
	///
	/// @return  
	///
	/// @par History:
	/// @li 6883/ZhongLiangJian, 2022/12/7-17:13:40
	///
	cv::Mat UnsharpenMask(const cv::Mat& input, int ksize = 25, float scale = 0.75);

	///
	/// @brief  伽马变换：增强图像亮度+对比度
	///     
	/// @param[in]  input  输入图像
	/// @param[in]  gamma  伽马值
	///
	/// @return  变换后的图像
	///
	/// @par History:
	/// @li 6883/ZhongLiangJian, 2023/3/24-9:24:44
	///
	cv::Mat GammaTransform(const cv::Mat& input, float gamma = 0.75);
};

#endif // _CV_TOOL_H_

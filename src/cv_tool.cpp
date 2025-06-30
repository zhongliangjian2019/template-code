#include <numeric>
#include "cv_tool.h"
#include "logger.h"

#define KERNELSIZE 5 // 滤波核大小
#define GRAYKERNELSIZE 9 // 计算液滴灰度时使用核大小，液滴取值半径为5个像素
#define DOUBLEEPSINON 0.0001  //double相等精度

///
/// @brief  获取最大轮廓索引
///     
/// @param[in] contours 输入轮廓数组
/// 
/// @return index 最大轮廓索引
///
/// @par History:
/// @li ZhongLiangJian, 2025/6/26-14:05:02
///
int CVTool::GetMaxContour(const std::vector<std::vector<cv::Point>>& contours)
{
	int max_idx = -1;
	// 一个元素都没有直接默认值
	// 只有一个元素
	if (contours.size() == 1)
	{
		max_idx = 0;
	}

	// 两个及以上
	if (contours.size() > 1)
	{
		std::vector<int> contour_areas;
		for (const auto& contour : contours)
		{
			contour_areas.push_back(cv::contourArea(contour));
		}
		max_idx = std::distance(contour_areas.begin(), std::max_element(contour_areas.begin(), contour_areas.end()));
	}

	return max_idx;
}

///
/// @brief  居中填充图像到方形
///     
/// @param[in] input 输入图像
/// @param[in] is_center 是否按中心填充
///	@param[out] pad_x 宽度方向填充量
/// @param[out] pad_y 高度方向填充量
/// 
/// @return  填充后的图像
///
/// @par History:
/// @li ZhongLiangJian, 2025/6/26-14:05:02
///
cv::Mat CVTool::FormatToSquare(const cv::Mat& input, bool is_center, int& pad_x, int& pad_y)
{
	int width = input.cols;
	int height = input.rows;

	int max_length = std::max(width, height);

	cv::Mat output = cv::Mat::zeros(cv::Size(max_length, max_length), input.type());

	if (is_center)
	{
		// 居中方式
		pad_x = (max_length - width) / 2;
		pad_y = (max_length - height) / 2;
		
	}
	else
	{
		// 左上方式
		pad_x = 0;
		pad_y = 0;
	}

	// 数据拷贝
	input.copyTo(output(cv::Rect(pad_x, pad_y, width, height)));
	return output;
}

///
/// @brief  计算轮廓中心（单个）
///     
/// @param[in]  contour 区域轮廓  
///
/// @return  轮廓中心点
///
/// @par History:
/// @li 6883/ZhongLiangJian, 2023/2/3-14:05:02
///
cv::Point CVTool::GetContourCenter(const std::vector<cv::Point> &contour)
{
    //LOG_DEBUG("Running ");
    cv::Moments mu;
    mu = cv::moments(contour, true);

    double centerX = 0;
    double centerY = 0;

    if((mu.m00 >= -DOUBLEEPSINON) && (mu.m00 <= DOUBLEEPSINON))
	{ 
		int size = contour.size();
        for (int i = 0; i < size; i++) 
		{
            centerX = centerX + contour[i].x; 
            centerY = centerY + contour[i].y;
        }
        centerX = centerX / size;
        centerY = centerY / size;
    }
    else
	{
        centerX = mu.m10 / mu.m00; 
        centerY = mu.m01 / mu.m00;
    }

    cv::Point mc(static_cast<int>(centerX), static_cast<int>(centerY));
    return mc;
}

///
/// @brief  计算轮廓中心（单个）和半径
///     
/// @param[in]  contour 区域轮廓  
/// @param[out]  center 轮廓中心
/// @param[out]  radius 轮廓半径
///
/// @return  None
///
/// @par History:
/// @li 6883/ZhongLiangJian, 2023/2/3-14:05:02
///
void CVTool::GetContourCenterAndRadius(const std::vector<cv::Point>& contour, cv::Point& center, int& radius, float scale)
{
    cv::Rect bndbox = cv::boundingRect(contour);
    center = cv::Point(bndbox.x + bndbox.width / 2, bndbox.y + bndbox.height / 2);
    radius = int((bndbox.width + bndbox.height) / (4.0 * scale));
}

///
/// @brief  计算轮廓质心（多个）
///     
/// @param[in]  contours  轮廓列表
/// @param[out]  centers  中心列表
///
/// @return  None
///
/// @par History:
/// @li 6883/ZhongLiangJian, 2023/2/3-14:11:44
///
void CVTool::GetContourCenter(const std::vector<std::vector<cv::Point>> &contours, std::vector<cv::Point> &centers)
{
	LOG_DEBUG("Running ");

    std::vector<cv::Point>().swap(centers);

    for (const auto& cnt: contours) 
	{
        cv::Point center = GetContourCenter(cnt);
        centers.push_back(center);
    }
}

/**
 * @brief 计算图像像素百分位数
 * @param[in] image: 输入图像
 * @param[in] percent: 分位点
 * @return value: 给定分位点对应的分位数
 * 
 * @li: zhongliangjian 2022/05/26 注释
 * 
 * @note:当前实现与numpy实现有差异, 结果可能相差一个灰度级
 */
float CVTool::ImagePercentile(cv::Mat& image, size_t percent)
{
	LOG_DEBUG("Running ");
	int histSize[] = { 256 };
	int channels[] = { 0 };
	float pranges[] = { 0, 256 };
	const float* ranges[] = { pranges };

	float loc = float(image.total() * (percent / 100.0));
	cv::Mat hist;
	cv::calcHist(&image, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);

	int value = 0;
	for (int count = 0; value < 256; ++value)
	{
		count += int(hist.at<float>(value));
		if (count >= loc) break;
	}
	return float(value);
}

/*
* 功能描述：移除图像盐噪声（由相机产生）
* param[in] input_image: 输入图像（灰度图）
* return dst_image: 去噪后的图像
* @li: zhongliangjian	2022/07/08	修改
*/
cv::Mat CVTool::RemoveImageNoise(const cv::Mat& input_image, int kernel_size)
{
	LOG_DEBUG("Running RemoveImageNoise(cv::Mat&)");

	// 检查输入是否为空
	if (input_image.empty())
	{
		LOG_WARNING("input image is empty");
		return cv::Mat();
	}

	// 输入图像通道数检查
	cv::Mat gray_image;
	if (input_image.channels() != 1)
	{
		LOG_WARNING("input image must is 1 channel, but give image channel is " + int2str(input_image.channels()));
		cv::cvtColor(input_image, gray_image, cv::COLOR_RGB2GRAY);
	}
	else
	{
		gray_image = input_image.clone();
	}

	//均值滤波
	cv::Mat blur_image;
    cv::medianBlur(gray_image, blur_image, kernel_size);
	//提取噪声
	gray_image.convertTo(gray_image, CV_32FC1);
	blur_image.convertTo(blur_image, CV_32FC1);

	cv::Mat noise_image;
	noise_image = gray_image - blur_image;

	// 将小于0的值置零避免转换uint8时反转
    noise_image = cv::abs(noise_image);
	noise_image.convertTo(noise_image, CV_8UC1);

	gray_image.convertTo(gray_image, CV_8UC1);
	blur_image.convertTo(blur_image, CV_8UC1);

	// 获取噪声掩膜
	cv::Mat thresh_image, none_noise_image, blur_noise_image;
	cv::threshold(noise_image, thresh_image, 15, 255, cv::THRESH_BINARY);

	cv::bitwise_and(blur_image, thresh_image, blur_noise_image);
	cv::bitwise_and(gray_image, 255 - thresh_image, none_noise_image);
	cv::Mat dst_image;
	dst_image = none_noise_image + blur_noise_image;

	return dst_image;
}

/**
* @brief 按中心裁剪图像
*
* @param[in] src_image 原图像
* @param[in] dsize 裁剪大小, 若dsize=(-1,-1)则表示按短边裁剪
*
* @return dst_image 裁剪后的图像
*
* @li: zhongliangjian 2022/09/26 编写
*/
cv::Mat CVTool::CenterCropImage(const cv::Mat& src_image, const cv::Size& dsize)
{
	LOG_DEBUG("Running ");

    cv::Range row_range(0, 0), col_range(0, 0);
    int row_crop = 0, col_crop = 0;
	cv::Mat dst_image;
    if (cv::Size(-1, -1) == dsize)
    {
        // 按短边裁剪
        int min_value = (src_image.rows > src_image.cols) ? src_image.cols : src_image.rows;
        row_crop = (src_image.rows - min_value) / 2;
        col_crop = (src_image.cols - min_value) / 2;
        row_range = cv::Range(row_crop, min_value + row_crop);
        col_range = cv::Range(col_crop, min_value + col_crop);
		dst_image = src_image(row_range, col_range).clone();
    }
    else
    {
        // 按给定大小裁剪
		dst_image = cv::Mat::zeros(dsize, src_image.type());
        row_crop = (src_image.rows - dsize.height) / 2;
        col_crop = (src_image.cols - dsize.width) / 2;

		if (row_crop >= 0 && col_crop >= 0)
		{
			row_range = cv::Range(row_crop, dsize.height + row_crop);
			col_range = cv::Range(col_crop, dsize.width + col_crop);
			dst_image = src_image(row_range, col_range).clone();
		}
		else if (row_crop < 0 && col_crop < 0)
		{
			src_image.copyTo(dst_image(cv::Range(-row_crop, src_image.rows - row_crop), cv::Range(-col_crop, src_image.cols - col_crop)));
		}
		else if (col_crop < 0)
		{
			row_range = cv::Range(row_crop, dsize.height + row_crop);
			col_range = cv::Range::all();
			src_image(row_range, col_range).copyTo(dst_image(cv::Range::all(), cv::Range(-col_crop, src_image.cols - col_crop)));
		}
		else
		{
			row_range = cv::Range::all();
			col_range = cv::Range(col_crop, dsize.width + col_crop);
			src_image(row_range, col_range).copyTo(dst_image(cv::Range(-row_crop, src_image.rows - row_crop), cv::Range::all()));
		}
    }

    return dst_image;
}

/**
* @brief 切分图像
*
* @param[in] srcImage 原图像
* @param[in] blockSize 分块图像大小
* @param[out] blockImages 分块后的图像
*
* @li: zhongliangjian 2022/09/21 编写
*/
void CVTool::GetSplitBlockImages(const cv::Mat& srcImage, const cv::Size& blockSize, std::vector<std::vector<cv::Mat>>& blockImages)
{
	LOG_DEBUG("Running ");

    blockImages.clear();

	// 计算分块数量
	int rowNum = static_cast<int>(ceil(srcImage.rows / double(blockSize.height)));
	int colNum = static_cast<int>(ceil(srcImage.cols / double(blockSize.width)));

	// 根据分块填充图像
	cv::Mat padImage = cv::Mat::zeros(cv::Size(colNum * blockSize.width, rowNum * blockSize.height), CV_8UC1);

	int rowPad = static_cast<int>(floor(padImage.rows - srcImage.rows) / 2);
	int colPad = static_cast<int>(floor(padImage.cols - srcImage.cols) / 2);

	srcImage.copyTo(padImage(cv::Range(rowPad, rowPad + srcImage.rows), cv::Range(colPad, colPad + srcImage.cols)));

	// 切分图像块
	for (int row = 0; row < rowNum; ++row)
	{
		cv::Range rowRange(row * blockSize.height, (row + 1) * blockSize.height);
		std::vector<cv::Mat> rowBlockImages;
		for (int col = 0; col < colNum; ++col)
		{
			cv::Range colRange(col * blockSize.width, (col + 1) * blockSize.width);
			rowBlockImages.push_back(padImage(rowRange, colRange).clone());
		}
		blockImages.push_back(rowBlockImages);
	}
}

///
/// @brief  获取图像ROI区域
///     
/// @param[in]  src  输入图像
/// @param[in]  center  中心位置
/// @param[in]  size  ROI区域大小（大于3的奇数）
///
/// @return  ROI图像
///
/// @par History:
/// @li 6883/ZhongLiangJian, 2023/2/4-10:59:21
///
cv::Mat CVTool::GetImageRoi(const cv::Mat& src, const cv::Point& center, int size)
{
	//LOG_DEBUG("Running ");

	if (size < 3 || size % 2 != 1)
	{
		return cv::Mat();
	}
	int offset = size / 2;
	cv::Range row_range(center.y - offset, center.y + offset + 1);
	cv::Range col_range(center.x - offset, center.x + offset + 1);
	if (row_range.start < 0 || row_range.end >= src.rows || col_range.start < 0 || col_range.end >= src.cols)
	{
		return cv::Mat();
	}
	else
	{
		return src(row_range, col_range);
	}
}

///
/// @brief  形态学膨胀重建
///     
/// @param[in]  temp  模板图像
/// @param[in]  mark  标记图像
/// @param[in]  connect  连通域决定核的形状
///
/// @return  
///
/// @par History:
/// @li 6883/ZhongLiangJian, 2023/2/4-11:38:00
///
cv::Mat CVTool::MorphRestructure(const cv::Mat& temp, const cv::Mat& mark, int connect)
{
	LOG_DEBUG("Running ");

	// 定义结构元素
	cv::Mat kernel;
	int shape = (connect == 4) ? cv::MORPH_CROSS : cv::MORPH_RECT;
	kernel = cv::getStructuringElement(shape, cv::Size(3, 3));

	// 膨胀重构
	cv::Mat dst = mark.clone();
	cv::Mat dilate;
	cv::Mat median;
	while (true)
	{
		cv::dilate(dst, dilate, kernel);
		cv::bitwise_and(dilate, temp, dilate);
		cv::bitwise_xor(dst, dilate, median);
		if (cv::countNonZero(dilate) == 0 || cv::countNonZero(median) == 0)
		{
			break;
		}
		dst = dilate.clone();
	}

	return dst;
}

///
/// @brief  形态学击中与击不中变换
///     
/// @param[in]  src  输入二值图像
/// @param[in]  kernel_1  前景匹配核
/// @param[in]  kernel_2  背景匹配核
/// @param[out]  dst  变换结果
///
/// @return  None
///
/// @par History:
/// @li 6883/ZhongLiangJian, 2023/2/4-13:09:45
///
cv::Mat CVTool::MorphHitAndMissTransform(const cv::Mat& src, const cv::Mat& kernel_1, const cv::Mat& kernel_2)
{
	LOG_DEBUG("Running ");

	// 前景匹配
	cv::Mat foreground = src.clone();
	cv::erode(foreground, foreground, kernel_1);

	// 背景匹配
	cv::Mat background = 255 - foreground;
	cv::erode(background, background, kernel_2);

	// 合并结果
	cv::Mat dst;
	cv::bitwise_and(foreground, background, dst);

	return dst;
}

///
/// @brief  形态学孔洞填充
///     
/// @param[in]  src  
/// @param[in]  connect  
///
/// @return  
///
/// @par History:
/// @li 6883/ZhongLiangJian, 2023/2/20-16:28:17
///
cv::Mat CVTool::MorphFillHole(const cv::Mat& src, int connect)
{
	// 模板图像
	cv::Mat fh_template = 255 - src;

	// 标记图像
	cv::Mat fh_mark = cv::Mat::zeros(fh_template.size(), CV_8UC1);
	fh_mark.row(0) = fh_template.row(0).clone();
	fh_mark.row(fh_mark.rows - 1) = fh_template.row(fh_template.rows-1).clone();
	fh_mark.col(0) = fh_template.col(0).clone();
	fh_mark.col(fh_mark.cols - 1) = fh_template.col(fh_template.cols - 1).clone();

	// 形态学重建
	cv::Mat dst = MorphRestructure(fh_template, fh_mark, connect);

	// 填充结果
	dst = 255 - dst;

	return dst;
}

///
/// @brief  形态学边界对象移除
///     
/// @param[in]  src  
/// @param[in]  connect  
///
/// @return  
///
/// @par History:
/// @li 6883/ZhongLiangJian, 2023/2/20-16:59:41
///
cv::Mat CVTool::MorphRemoveEdgeObject(const cv::Mat & src, int connect)
{
	LOG_DEBUG("Running ");

	// 模版图像
	cv::Mat reo_template = src.clone();

	// 标记图像
	cv::Mat reo_mark = cv::Mat::zeros(reo_template.size(), CV_8UC1);
	reo_template.row(0).copyTo(reo_mark.row(0));
	reo_template.row(reo_template.rows - 1).copyTo(reo_mark.row(reo_mark.rows - 1));
	reo_template.col(0).copyTo(reo_mark.col(0));
	reo_template.col(reo_template.cols - 1).copyTo(reo_mark.col(reo_mark.cols - 1));

	// 形态学重建边界对象
	cv::Mat dst = MorphRestructure(reo_template, reo_mark, connect);

	// 去除边界对象
	dst = reo_template - dst;

	return dst;
}

///
/// @brief  获取二值图像的轮廓质心
///     
/// @param[in]  src  二值图像
/// @param[in]  centers  输出中心
/// @param[in]  area_min  轮廓面积最小值,小于该值的轮廓无效
/// @param[in]  area_max  轮廓面积最大值,大于该值的轮廓无效
///
/// @return  
///
/// @par History:
/// @li 6883/ZhongLiangJian, 2023/2/10-10:21:04
///
void CVTool::GetBinaryContourCenter(const cv::Mat& src, std::vector<cv::Point>& centers, int area_min, int area_max)
{
	LOG_DEBUG("Running ");

	// 轮廓检测
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(src, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	// 筛选符合要求的轮廓
	std::vector<std::vector<cv::Point2i>> valid_contours;
	//std::vector<double> area_vec;
	for (auto contour : contours)
	{
		double cnt_area = cv::contourArea(contour);
		//area_vec.push_back(cnt_area);
		if (area_min < cnt_area && cnt_area < area_max)
		{
			valid_contours.push_back(contour);
		}
	}

	// 获取轮廓中心位置
	centers.clear();
	CVTool::GetContourCenter(contours, centers);
}

///
/// @brief  按给定点平移图像
///     
/// @param[in]  srcImage  输入图像
/// @param[in]  offset  x,y方向平移量
///
/// @return  平移后的图像
///
/// @par History:
/// @li 6883/ZhongLiangJian, 2023/2/13-10:35:45
///
cv::Mat CVTool::MoveImage(const cv::Mat& srcImage, const cv::Point& offset)
{
	LOG_DEBUG("Runing ");
	cv::Mat dstImage = cv::Mat::zeros(cv::Size(srcImage.cols, srcImage.rows), srcImage.type());
	cv::Mat tempImage = srcImage.clone();

	// 平移x方向
	if (offset.x > 0)
	{
		tempImage(cv::Range(0, tempImage.rows), cv::Range(0, tempImage.cols - offset.x)).copyTo(dstImage(cv::Range(0, dstImage.rows), cv::Range(offset.x, dstImage.cols)));
	}
	else if (offset.x < 0)
	{
		tempImage(cv::Range(0, tempImage.rows), cv::Range(-offset.x, tempImage.cols)).copyTo(dstImage(cv::Range(0, dstImage.rows), cv::Range(0, dstImage.cols + offset.x)));
	}
	else
	{
		tempImage.copyTo(dstImage);
	}

	tempImage = dstImage.clone();
	dstImage = cv::Scalar::all(0);

	// 平移y方向
	if (offset.y > 0)
	{
		tempImage(cv::Range(0, tempImage.rows - offset.y), cv::Range(0, tempImage.cols)).copyTo(dstImage(cv::Range(offset.y, dstImage.rows), cv::Range(0, dstImage.cols)));
	}
	else if (offset.y < 0)
	{
		tempImage(cv::Range(-offset.y, tempImage.rows), cv::Range(0, tempImage.cols)).copyTo(dstImage(cv::Range(0, dstImage.rows + offset.y), cv::Range(0, dstImage.cols)));
	}
	else
	{
		tempImage.copyTo(dstImage);
	}

	return dstImage;
}

///
/// @brief  图像平移
///     
/// @param[in]  srcImages  输入图像列表
/// @param[in]  offsets  平移量列表
/// @param[in]  dstImages  平移图像列表
///
/// @return  None
///
/// @par History:
/// @li 6883/ZhongLiangJian, 2023/2/13-10:45:31
///
void CVTool::MoveImages(const std::vector<cv::Mat>& srcImages, const std::vector<cv::Point>& offsets, std::vector<cv::Mat>& dstImages)
{
	LOG_DEBUG("Running ");

	dstImages.clear();

	if (srcImages.size() != offsets.size())
	{
		dstImages = srcImages;
		return;
	}

	for (size_t i = 0; i < srcImages.size(); ++i)
	{
		dstImages.push_back(MoveImage(srcImages.at(i), offsets.at(i)));
	}
}


///
/// @brief  图像分块
///     
/// @param[in]  src  原始图像
/// @param[out]  block_images 分块图像 
/// @param[out]  block_offsets 分块图像位置偏移量
/// @param[out]  pad_image_size 分块时进行填充后的图像大小, 用于拼接复原
/// @param[out]  valid_size  分块有效区域大小, 用于拼接复原
/// @param[out]  offset  分块起始偏移量, 用于拼接复原
/// @param[in]  block_size  分块大小
/// @param[in]  overlap_ratio  分块重叠率
///
/// @return  
///
/// @par History:
/// @li 6883/ZhongLiangJian, 2023/2/21-16:34:19
///
void CVTool::SplitImageBlock(const cv::Mat& src, std::vector<cv::Mat>& block_images, std::vector<cv::Point>& block_offsets,
	cv::Size& pad_image_size, const cv::Size& block_size, float overlap_ratio)
{
	LOG_DEBUG("Running");

	cv::Mat image = src.clone();
	// 计算重叠信息
	cv::Size overlap = cv::Size(int(block_size.width * overlap_ratio / 2) * 2, int(block_size.height * overlap_ratio / 2) * 2);
	cv::Size valid_size = cv::Size(block_size.width - overlap.width, block_size.height - overlap.height);
	cv::Size offset = cv::Size(overlap.width / 2, overlap.height / 2);

	// 计算图像分块数
	int block_rows = image.rows / valid_size.height;
	int block_cols = image.cols / valid_size.width;

	// 不能完整分块,进行填充处理
	if (image.rows % valid_size.height != 0)
	{
		block_rows += 1;
	}
	if (image.cols % valid_size.width != 0)
	{
		block_cols += 1;
	}

	// 计算分块填充图像的大小
	pad_image_size.width = block_cols * valid_size.width + overlap.width;
	pad_image_size.height = block_rows * valid_size.height + overlap.height;

	// 生成填充图像
	cv::Mat pad_image = cv::Mat::zeros(pad_image_size, image.type());
	image.copyTo(pad_image(cv::Range(offset.height, offset.height + image.rows), cv::Range(offset.width, offset.width + image.cols)));
	// 图像分块
	block_images.clear();
	block_offsets.clear();
	for (int row = 0; row < block_rows; ++row)
	{
		for (int col = 0; col < block_cols; ++col)
		{
			cv::Range row_range = cv::Range(row * valid_size.height, row * valid_size.height + block_size.height);
			cv::Range col_range = cv::Range(col * valid_size.width, col * valid_size.width + block_size.width);
			block_images.push_back(pad_image(row_range, col_range).clone());
			block_offsets.push_back(cv::Point(col_range.start, row_range.start));
		}
	}
}

///
/// @brief  图像分块
///     
/// @param[in]  src  原始图像
/// @param[out]  block_images 分块图像 
/// @param[out]  block_offsets 分块图像位置偏移量
/// @param[out]  pad_image_size 分块时进行填充后的图像大小, 用于拼接复原
/// @param[out]  valid_size  分块有效区域大小, 用于拼接复原
/// @param[out]  offset  分块起始偏移量, 用于拼接复原
/// @param[in]  block_size  分块大小
/// @param[in]  overlap_ratio  分块重叠率
///
/// @return  
///
/// @par History:
/// @li 6883/ZhongLiangJian, 2023/2/21-16:34:19
///
void CVTool::SplitImageBlockA(const cv::Mat& src, std::vector<cv::Mat>& block_images, std::vector<cv::Point>& block_offsets,
	const cv::Size& block_size, float overlap_ratio)
{
	LOG_DEBUG("Running");

	cv::Mat image = src;
	// 计算重叠信息
	cv::Size overlap = cv::Size(int(block_size.width * overlap_ratio / 2) * 2, int(block_size.height * overlap_ratio / 2) * 2);
	cv::Size valid_size = cv::Size(block_size.width - overlap.width, block_size.height - overlap.height);

	// 计算图像分块数
	int block_rows = image.rows / valid_size.height;
	int block_cols = image.cols / valid_size.width;

	// 不能完整分块,进行填充处理
	if (image.rows % valid_size.height != 0)
	{
		block_rows += 1;
	}
	if (image.cols % valid_size.width != 0)
	{
		block_cols += 1;
	}

	// 确定有效分块大小
	valid_size.height = block_size.height - int((block_rows * block_size.height - image.rows) / (double)(block_rows - 1) + 1);
	valid_size.width = block_size.width - int((block_cols * block_size.width - image.cols) / (double)(block_cols - 1) + 1);

	// 图像分块
	block_images.clear();
	block_offsets.clear();
	for (int row = 0; row < block_rows; ++row)
	{
		for (int col = 0; col < block_cols; ++col)
		{
			cv::Range row_range = cv::Range(row * valid_size.height, row * valid_size.height + block_size.height);
			cv::Range col_range = cv::Range(col * valid_size.width, col * valid_size.width + block_size.width);
			//std::cout << row << " " << col << " " << row_range.start << " " << row_range.end << " " << col_range.start << " " << col_range.end << std::endl;
			block_images.push_back(image(row_range, col_range).clone());
			block_offsets.push_back(cv::Point(col_range.start, row_range.start));
		}
	}
}


///
/// @brief  拼接图像块, 为 SplitImageBlock()函数的逆操作
///     
/// @param[in]  block_images  图像块
/// @param[in]  block_offsets  图像块位置偏移
/// @param[in]  offset  基准偏移
/// @param[in]  valid_size  
/// @param[in]  pad_size  
/// @param[in]  src_size  
///
/// @return  
///
/// @par History:
/// @li 6883/ZhongLiangJian, 2023/2/21-16:59:31
///
cv::Mat CVTool::MosaicImageBlock(const std::vector<cv::Mat>& block_images, const std::vector<cv::Point>& block_offsets, const cv::Size& pad_size, const cv::Size& src_size, float overlap_ratio)
{
	LOG_DEBUG("Running ");

	cv::Size block_size = cv::Size(block_images.at(0).cols, block_images.at(0).rows);
	int mat_type = block_images.at(0).type();
	cv::Size overlap = cv::Size(int(block_size.width * overlap_ratio / 2) * 2, int(block_size.height * overlap_ratio / 2) * 2);
	cv::Size valid_size = cv::Size(block_size.width - overlap.width, block_size.height - overlap.height);
	cv::Size offset = cv::Size(overlap.width / 2, overlap.height / 2);


	// 复原到分块前的填充图像
	cv::Mat pad_image = cv::Mat::zeros(pad_size, mat_type);
	for (size_t i = 0; i < block_offsets.size(); ++i)
	{
		cv::Range src_row_range = cv::Range(offset.height, offset.height + valid_size.height);
		cv::Range src_col_range = cv::Range(offset.width, offset.width + valid_size.width);
		cv::Range dst_row_range = cv::Range(block_offsets[i].y + offset.height, (block_offsets[i].y + valid_size.height + offset.height));
		cv::Range dst_col_range = cv::Range(block_offsets[i].x + offset.width, (block_offsets[i].x + valid_size.width + offset.width));
		block_images.at(i)(src_row_range, src_col_range).copyTo(pad_image(dst_row_range, dst_col_range));
	}

	// 从填充图像中截取原图
	cv::Mat dst = pad_image(cv::Range(offset.height, offset.height + src_size.height), cv::Range(offset.width, offset.width + src_size.width)).clone();

	return dst;
}

///
/// @brief  读取图像
///     
/// @param[in]  image_path  图像文件路径
///
/// @return  读取的图像
///
/// @par History:
/// @li 6883/ZhongLiangJian, 2023/2/21-16:59:31
///
cv::Mat CVTool::ReadImage(const std::string& image_path, int flag)
{
	cv::Mat image = cv::imread(image_path, flag);
	return image;
}

///
/// @brief  保存图像
///     
/// @param[in]  image_paths  图像文件路径
/// @param[in]  dst_size  默认裁剪后图像大小
///
/// @return  裁剪后的图像
///
/// @par History:
/// @li 6883/ZhongLiangJian, 2023/2/21-16:59:31
///
void CVTool::SaveImage(const cv::Mat& image, const std::string& save_path)
{
	LOG_DEBUG("Runing ");
	cv::imwrite(save_path, image);
}

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
cv::Mat CVTool::LogTransform(const cv::Mat& input, int light_min, int light_max, float drop_ratio)
{
	LOG_DEBUG("Running");

	cv::Mat src = input.clone();

	// 提取微滴区域最小值
	int row = src.rows / 2;
	int col = src.cols / 2;
	cv::Range row_range(row - int(src.rows * drop_ratio), row + int(src.rows * drop_ratio));
	cv::Range col_range(col - int(src.cols * drop_ratio), col + int(src.cols * drop_ratio));
	cv::Mat roi = src(row_range, col_range).clone();	
	double min_value = 0;
	cv::minMaxIdx(roi, &min_value);

	// 提取全局最大值
	cv::Mat dst;
	src.convertTo(dst, CV_32F);
	double max_value = 0;
	cv::minMaxIdx(dst, 0, &max_value);

	// 归一化
	dst = (dst - min_value) / (max_value - min_value) + 1;
	cv::Mat mark = (dst > 1) / 255;
	mark.convertTo(mark, CV_32F);
	dst = dst.mul(mark) * 1.2 + dst.mul(1 - mark);

	// 对数变换
	cv::log(dst, dst);

	// 背景掩膜
	cv::Mat mask = (dst < 0) / 255;
	mask.convertTo(mask, CV_32F);

	// 前景处理
	cv::Mat foreground = dst.clone();
	foreground = foreground.mul(1 - mask);
	cv::normalize(foreground, foreground, 0, light_max, cv::NORM_MINMAX);

	// 背景处理
	cv::Mat background = dst.clone();
	background = background.mul(mask);
	cv::normalize(background, background, 0, light_min, cv::NORM_MINMAX);

	// 混合结果
	cv::Mat result = foreground + background;
	result.convertTo(result, CV_8U);

	return result;
}

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
cv::Mat CVTool::GammaTransform(const cv::Mat& input, float gamma)
{
	LOG_DEBUG("Running");

	// 屏蔽机制
	cv::Mat image = input.clone();
	if (image.channels() != 1)
	{
		if (image.channels() == 3)
		{
			cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
		}
		else
		{
			cv::cvtColor(image, image, cv::COLOR_BGRA2GRAY);
		}
	}

	// 构建查找表
	cv::Mat gamma_table = cv::Mat::zeros(cv::Size(256, 1), CV_32F);
	float* row_ptr = gamma_table.ptr<float>(0);
	for (int i = 0; i < 256; ++i)
	{
		row_ptr[i] = i;
	}
	gamma_table /= 255.0;
	cv::pow(gamma_table, gamma, gamma_table);
	gamma_table.convertTo(gamma_table, CV_8U, 255);

	// 查找表变换
	cv::Mat dst;
	cv::LUT(input, gamma_table, dst);

	return dst;
}

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
cv::Mat CVTool::UnsharpenMask(const cv::Mat& input, int ksize, float scale)
{
	LOG_DEBUG("Running");

	cv::Mat src = input.clone();
	cv::Mat gaussion;
	cv::GaussianBlur(src, gaussion, cv::Size(ksize, ksize), ksize / 6.0);
	
	src.convertTo(src, CV_32FC1);
	gaussion.convertTo(gaussion, CV_32FC1);
	cv::Mat diff = src - gaussion;
	cv::threshold(diff, diff, 0, 1, cv::THRESH_TOZERO);

	cv::Mat dst = src + diff / scale;
	cv::threshold(dst, dst, 255, 255, cv::THRESH_TRUNC);

	dst.convertTo(dst, CV_8UC1);

	return dst;
}

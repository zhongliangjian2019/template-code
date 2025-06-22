///
/// 自定义通用工具函数
///

#ifndef _COM_TOOL_H_
#define _COM_TOOL_H_

#include <algorithm>
#include <numeric>
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include "logger.h"

#define T_PI						3.141592653589793	// 圆周率

/**
* @brief 求和、均值、标准差
*/
template <typename T>
void GetSumAndMeanAndSTD(const std::vector<T>& data, double* sum = nullptr, double* mean = nullptr, double* std = nullptr)
{
	LOG_DEBUG("Runing ");
	if (nullptr != sum)
	{
		*sum = std::accumulate(data.begin(), data.end(), 0.0);
	}
	
	if (nullptr != mean)
	{
		*mean = std::accumulate(data.begin(), data.end(), 0.0) / double(data.size());
	}

	if (nullptr != std)
	{
		double mean_value = std::accumulate(data.begin(), data.end(), 0.0) / double(data.size());
		double sum_value = 0.0;
		std::for_each(data.begin(), data.end(), [&](const T& value) {sum_value += (value - mean_value) * (value - mean_value); });
		*std = std::sqrt(sum_value / data.size());
	}
}

/**
* @brief 高斯函数
* 
* @param[in] x 输入
* @param[in] mean 均值
* @param[in] sigma 标准差
* 
* return 函数值
*/
template <typename T>
double Gaussion(const T x, double mean = 0.0, double sigma = 1.0)
{
	static double gauss_c = 1.0 / (sigma * std::sqrt(2 * T_PI));
	return gauss_c * std::exp(-0.5 * std::pow((x - mean) / sigma, 2));
}

/**
* @brief 高斯滤波
* 
* @param[in] input 输入数据
* @param[out] output 滤波输出数据
* @param[in] ksize 滤波器窗口大小
* @param[in] sigma 高斯核的标准差
* 
* @li zhongliangjian 2022/10/12 编写
*/
template <typename T>
void GaussionBlur(const std::vector<T>& input, std::vector<T>& output, int ksize = 3, double sigma = 2.5)
{
	LOG_DEBUG("Runing ");
	cv::Mat kernel = cv::getGaussianKernel(ksize, sigma);
	kernel = kernel / cv::sum(kernel)[0];
	int radius = ksize / 2;
	output.clear();
	for (size_t i = 0; i < input.size(); ++i)
	{
		if (i < radius || i >= input.size() - radius)
		{
			output.push_back(input.at(i));
			continue;
		}

		double sum_blur = 0.0;
		for (int j = 0; j < ksize; ++j)
		{
			sum_blur += input.at(i - radius + j) * kernel.at<double>(j);
		}

		output.push_back(static_cast<T>(sum_blur));
	}
}

///
/// @brief  排序比较函数(v1 < v2 升序, v1 > v2 降序)
///     
/// @param[in]  v1  比较值1
/// @param[in]  v2  比较值2
///
/// @return  比较结构
///
/// @par History:
/// @li 6883/ZhongLiangJian, 2023/2/1-16:41:15
///
template <typename T>
bool mycompare(T v1, T v2)
{
	return v1 < v2;
}

// 获取局部最大值
template <typename T>
void local_maxima_1d(const std::vector<T>& x, std::vector<int>& points)
{
	LOG_DEBUG("Runing ");
	points.clear();

	size_t i = 1;
	size_t i_max = x.size() - 1;
	size_t i_ahead = 0;
	while (i < i_max)
	{
		if (x[i - 1] < x[i])
		{
			i_ahead = i + 1;
			while (i_ahead < i_max && x[i_ahead] == x[i])
			{
				i_ahead += 1;
			}
			if (x[i_ahead] < x[i])
			{
				points.push_back((i + i_ahead - 1) / 2);
				i = i_ahead;
			}
		}
		i += 1;
	}
}

// 获取局部最小值
template <typename T>
void local_minima_1d(const std::vector<T>& x_vec, std::vector<int>& point_vec)
{
	LOG_DEBUG("Runing ");
	std::vector<T> curr_x_vec;
	for (auto v : x_vec)
	{
		curr_x_vec.push_back(-v);
	}
	local_maxima_1d(curr_x_vec, point_vec);
}

template <typename T>
void find_valley_points(const std::vector<T>& x, std::vector<PeakInfo>& peak_infos)
{
	LOG_DEBUG("Runing ");
	// 没有峰直接退出
	if (peak_infos.size() == 0)
	{
		return;
	}
	// 获取第一个峰的左谷点
	std::vector<int> potential_valley_points;
	for (int i = peak_infos[0].peak_point - 1; i > 0; --i)
	{
		if (x[i] < x[i + 1] && x[i] == x[i - 1])
		{
			potential_valley_points.push_back(i);
		}
	}
	int valley_point = 0;
	if (potential_valley_points.size() != 0)
	{
		valley_point = potential_valley_points.back();
	}
	peak_infos[0].left_point = valley_point;

	// 获取最后一个峰的右谷点
	potential_valley_points.clear();
	for (int i = peak_infos[peak_infos.size() - 1].peak_point + 1; i < peak_infos.size() - 1; ++i)
	{
		if (x[i] < x[i - 1] && x[i] == x[i + 1])
		{
			potential_valley_points.push_back(i);
		}
	}
	valley_point = x.size() - 1;
	if (potential_valley_points.size() != 0)
	{
		valley_point = potential_valley_points.back();
	}
	peak_infos[peak_infos.size() - 1].right_point = valley_point;

	// 获取峰间谷点
	if (peak_infos.size() > 1)
	{
		// 寻找两个峰之间的谷点
		for (size_t i = 0; i < peak_infos.size() - 1; ++i)
		{
			auto min_elem = std::min_element(x.begin() + peak_infos[i].peak_point, x.begin() + peak_infos[i + 1].peak_point);
			valley_point = std::distance(x.begin(), min_elem);
			peak_infos[i].right_point = valley_point;
			peak_infos[i + 1].left_point = valley_point;
		}
	}
}

// 峰查找
template <typename T>
void find_peaks(const std::vector<T>& x, std::vector<PeakInfo>& peak_infos, int max_peak = 2, float height = 0, float width = 0, float area = 0, float distance = 0)
{
	LOG_DEBUG("Runing ");
	// 获取峰点
	std::vector<int> peak_points;
	local_maxima_1d(x, peak_points);

	LOG_INFO("peak point num: " + std::to_string(peak_points.size()));

	if (peak_points.size() == 0)
	{
		return;
	}

	// 初始化峰信息
	peak_infos.clear();
	for (auto peak : peak_points)
	{
		peak_infos.push_back(PeakInfo{ peak, -1, -1, x[peak] });
	}

	// 获取峰谷点信息
	find_valley_points(x, peak_infos);

	// 仅有一个峰，不做筛选处理
	if (peak_points.size() <= max_peak)
	{
		return;
	}

	// 条件筛选
	std::vector<PeakInfo> curr_peak_infos;
	curr_peak_infos.swap(peak_infos);
	int count = curr_peak_infos.size();
	// 高度筛选
	for (auto peak : curr_peak_infos)
	{
		count -= 1;
		if ((count + peak_infos.size() < max_peak) || (peak.peak_height > height))
		{
			peak_infos.push_back(peak);
		}
	}

	// 宽度筛选
	curr_peak_infos.swap(peak_infos);
	peak_infos.clear();
	count = curr_peak_infos.size();
	for (auto peak : curr_peak_infos)
	{
		count -= 1;
		if ((count + peak_infos.size() < max_peak) || (peak.right_point - peak.left_point > width))
		{
			peak_infos.push_back(peak);
		}
	}

	// 面积赛选
	curr_peak_infos.swap(peak_infos);
	peak_infos.clear();
	count = curr_peak_infos.size();
	for (auto peak : curr_peak_infos)
	{
		count -= 1;
		if ((count + peak_infos.size() < max_peak) || (std::accumulate(x.begin() + peak.left_point, x.begin() + peak.right_point, 0.0) > area))
		{
			peak_infos.push_back(peak);
		}
	}

	// 峰距离筛选
	curr_peak_infos.swap(peak_infos);
	peak_infos.clear();
	int i = 0;
	while (i < curr_peak_infos.size() - 1)
	{
		if (curr_peak_infos.size() + peak_infos.size() - i - 1 < max_peak)
		{
			peak_infos.push_back(curr_peak_infos[i]);
			i += 1;
			continue;
		}

		if (curr_peak_infos[i + 1].peak_point - curr_peak_infos[i].peak_point < distance)
		{
			peak_infos.push_back(x[curr_peak_infos[i + 1].peak_point] > x[curr_peak_infos[i].peak_point] ? curr_peak_infos[i + 1] : curr_peak_infos[i]);
			i += 2;
		}
		else
		{
			peak_infos.push_back(curr_peak_infos[i]);
			i += 1;
		}
	}

	if (peak_infos.size() == 0)
	{
		peak_infos.swap(curr_peak_infos);
	}
	else
	{
		if (curr_peak_infos.back().peak_point - peak_infos.back().peak_point > distance)
		{
			peak_infos.push_back(curr_peak_infos.back());
		}
	}
	// 提取最高的前max_peak个峰
	if (peak_infos.size() > max_peak)
	{
		for (auto& peak : peak_infos)
		{
			peak.peak_height = x[peak.peak_point];
		}
		curr_peak_infos.swap(peak_infos);
		peak_infos.clear();
		std::sort(curr_peak_infos.begin(), curr_peak_infos.end(),
			[](const PeakInfo& peak1, const PeakInfo& peak2) {return peak1.peak_height < peak2.peak_height; });
		for (int i = 0; i < max_peak; ++i)
		{
			peak_infos.push_back(curr_peak_infos.back());
			curr_peak_infos.pop_back();
		}
		std::sort(peak_infos.begin(), peak_infos.end(),
			[](const PeakInfo& peak1, const PeakInfo& peak2) {return peak1.peak_point < peak2.peak_point; });
	}

	// 对筛选后的峰更新谷点
	find_valley_points(x, peak_infos);
}

///
/// @brief  多项式拟合
///     
/// @param[in]  point_vec	拟合数据坐标
/// @param[in]  x_vec		待拟合数据点横坐标
/// @param[in]  y_vec		拟合出的纵坐标
/// @param[in]  n			多项式次数
///
/// @return  成功/失败
///
/// @par History:
/// @li 6883/ZhongLiangJian, 2023/3/2-13:56:00
///
bool GetPolyFitResult(const std::vector<cv::Point>& point_vec, const std::vector<float>& x_vec, std::vector<float>& y_vec, int n)
{
	LOG_DEBUG("Running ");

	int N = point_vec.size();

	// 构建系数矩阵X,Y
	cv::Mat x_v = cv::Mat::zeros(N, 1, CV_64F);
	cv::Mat Y = x_v.clone();

	cv::Point point;
	for (int i = 0; i < N; ++i)
	{
		point = point_vec.at(i);
		*x_v.ptr<double>(i, 0) = static_cast<double>(point.x);
		*Y.ptr<double>(i, 0) = static_cast<double>(point.y);
	}

	// 归一化
	double xmax = 0, ymax = 0;
	cv::minMaxIdx(x_v, 0, &xmax);
	cv::minMaxIdx(Y, 0, &ymax);

	x_v /= xmax;
	Y /= ymax;

	cv::Mat X = cv::Mat::ones(N, 1, CV_64F);
	for (int i = 0; i < n; ++i)
	{
		if (i == 0)
		{
			cv::hconcat(X, x_v, X);
		}
		else
		{
			x_v = x_v.mul(x_v);
			cv::hconcat(X, x_v, X);
		}

	}

	// 求解
	cv::Mat A = cv::Mat::zeros(n + 1, 1, CV_64F);
	if (!cv::solve(X, Y, A, cv::DECOMP_NORMAL))
	{
		return false;
	}
	else
	{
		// 构建输出系数矩阵
		cv::Mat X_out = cv::Mat::ones(x_vec.size(), 1, CV_64F);
		cv::Mat x_vout = cv::Mat(x_vec, true);
		x_vout.convertTo(x_vout, CV_64F);
		x_vout = x_vout.reshape(1, x_vec.size());
		x_vout /= xmax;
		for (int i = 0; i < n; ++i)
		{
			if (i == 0)
			{
				cv::hconcat(X_out, x_vout, X_out);
			}
			else
			{
				x_vout = x_vout.mul(x_vout);
				cv::hconcat(X_out, x_vout, X_out);
			}
		}
		cv::Mat Y_out = X_out * A;
		Y_out *= ymax;
		y_vec.clear();
		for (int i = 0; i < Y_out.rows; ++i)
		{
			y_vec.push_back(*Y_out.ptr<double>(i, 0));
		}
		return true;
	}
}

// 峰信息数组
struct PeakInfo
{
	int peak_point = 0;
	int left_point = 0;
	int right_point = 0;
	float peak_height = 0;
};

// 数据归一化
void DataNormalize(std::vector<double>& data, double out_max, double out_min = 0, double in_max = -1, double in_min = -1);

///
/// @brief  判断文件是否存在
///     
/// @param[in]  filename  图像文件路径
///
/// @return  true/false
///
/// @par History:
/// @li 6883/ZhongLiangJian, 2023/5/12-17:31
///
bool IsFileExists(const std::string& filename);

// 从路径中分离文件名
std::string GetFilename(const std::string& path);

///
/// @brief  windows local string to unicode8 string
///     
/// @param[in]  windows local string
///
/// @return  unicode8 string
///
/// @par History:
/// @li 6883/ZhongLiangJian, 2023/5/12-17:31
///
std::string localToUtf8(const std::string& str);

// 字符串替换函数
std::string StrReplace(const std::string& strSrc, const std::string& oldStr, const std::string& newStr, int count = -1);

#endif // _COM_TOOL_H_
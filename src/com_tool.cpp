#include <vector>
#include <algorithm>
#include <sys/stat.h>
#include <iomanip>
#include <sstream>
#include "com_tool.h"

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
bool IsFileExists(const std::string& filename)
{
	struct stat buffer;
	return (stat(filename.c_str(), &buffer) == 0);
}

///
/// @brief  从路径中获取文件名
///     
/// @param[in]	path  文件路径  
///
/// @return  文件名
///
/// @par History:
/// @li 6883/ZhongLiangJian, 2023/2/20-11:49:42
///
std::string GetFilename(const std::string& path)
{
	size_t pos = path.find_last_of("/\\");
	if (pos != std::string::npos)
	{
		return path.substr(pos + 1);
	}
	else
	{
		return path;
	}
}

///
/// @brief  数据归一化
///     
/// @param[in/out]  data	输入数据  
/// @param[in]		out_max	输出最大值  
/// @param[in]		out_min 输出最小值
/// @param[in]		in_min  输入最小值
/// @param[in]		in_max  输入最大值
///
/// @return  
///
/// @par History:
/// @li 6883/ZhongLiangJian, 2023/2/20-11:49:42
///
void DataNormalize(std::vector<double>& data, double out_max, double out_min, double in_max, double in_min)
{
	if (-1 == in_min)
	{
		in_min = *(std::min_element(data.begin(), data.end()));
	}
	if (-1 == in_max)
	{
		in_max = *(std::max_element(data.begin(), data.end()));
	}

	cv::Mat data_mat = cv::Mat(data);
	data_mat = (data_mat - in_min) / (in_max - in_min + 1) * (out_max - out_min) + out_min;
}

///
/// @brief  字符串替换
///     
/// @param[in]  strSrc  字符串数据
/// @param[in]  oldStr  旧字符串
/// @param[in]  newStr  新字符串
///
/// @return  
///
/// @par History:
/// @li 6883/ZhongLiangJian, 2023/5/12-17:31
///
std::string StrReplace(const std::string& strSrc, const std::string& oldStr, const std::string& newStr, int count)
{
	std::string strRet = strSrc;
	size_t pos = 0;
	int cur_count = 0;
	if (-1 == count)
	{
		count = strRet.size();
	}
	while ((pos = strRet.find(oldStr, pos)) != std::string::npos)
	{
		strRet.replace(pos, oldStr.size(), newStr);
		if (++cur_count >= count)
		{
			break;
		}
		pos += newStr.size();
	}
	return strRet;
}
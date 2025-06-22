///
/// 基于log4cpp的自定义日志记录器
///

#ifndef _LOGGER_H_
#define _LOGGER_H_

#include <iostream>
#include <string>
#include <log4cplus/logger.h>
#include <log4cplus/configurator.h>
#include <log4cplus/helpers/stringhelper.h>
#include <log4cplus/loggingmacros.h>
#include <stdio.h>


// 日志配置文件路径
#define LOG_CONF_PATH ".\\AlgorithmLog.cfg"

#ifdef WIN32
#define __FILENAME__ (strrchr(__FILE__, '\\') ? (strrchr(__FILE__, '\\') + 1) : __FILE__)
#else
#define __FILENAME__ (strrchr(__FILE__, '/') ? (strrchr(__FILE__, '/') + 1) : __FILE__)
#endif

#define LOG_MSG(msg) \
		std::string(msg)\
		.append(" local: ")\
        .append(__FILENAME__)\
		.append(" -> ")\
		.append(__func__)\
        .append("() -> line (")\
		.append(int2str(__LINE__))\
        .append(") ")\
		.c_str()

#define LOG_ERROR(msg)   AlgorithmLog::GetInstance().ErrorLog(LOG_MSG(msg))
#define LOG_WARNING(msg) AlgorithmLog::GetInstance().WarnLog(LOG_MSG(msg))
#define LOG_INFO(msg)    AlgorithmLog::GetInstance().InfoLog(LOG_MSG(msg))
#define LOG_DEBUG(msg)   AlgorithmLog::GetInstance().DebugLog(LOG_MSG(msg))

// 转换整数到字符串
inline std::string int2str(int iFileLine)
{
	std::ostringstream oss;
	oss << iFileLine;
	return oss.str();
}

// 转换浮点是到字符串：双精度
inline std::string double2str(double value)
{
	std::ostringstream oss;
	oss.precision(4);
	oss << value;
	return oss.str();
}

// 转换浮点是到字符串：单精度
inline std::string float2str(float value)
{
	return double2str(double(value));
}

// 算法日志单例模式类
class AlgorithmLog
{
private:
	// 构造函数私有
	AlgorithmLog();

public:
	// 禁用复制/拷贝构造函数
	AlgorithmLog(AlgorithmLog&) = delete;
	AlgorithmLog& operator=(const AlgorithmLog&) = delete;

	// 静态成员函数获取唯一实例
	static AlgorithmLog& GetInstance();

	// 输出各级别日志的函数
	void ErrorLog(const char* pInfo);
	void WarnLog(const char* pInfo);
	void InfoLog(const char* pInfo);
	void DebugLog(const char* pInfo);


private:
	log4cplus::Logger m_logger;
};

#endif // _LOGGER_H_


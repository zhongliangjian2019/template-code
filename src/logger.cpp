#include "logger.h"

AlgorithmLog::AlgorithmLog()
{
	log4cplus::PropertyConfigurator::doConfigure(LOG4CPLUS_TEXT(LOG_CONF_PATH));
	m_logger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("AlgorithmLog"));
}

AlgorithmLog& AlgorithmLog::GetInstance()
{
	static AlgorithmLog instance;
	return instance;
}

void AlgorithmLog::ErrorLog(const char* pInfo)
{
	LOG4CPLUS_ERROR(m_logger, pInfo);
}

void AlgorithmLog::WarnLog(const char* pInfo)
{
	LOG4CPLUS_WARN(m_logger, pInfo);
}

void AlgorithmLog::InfoLog(const char* pInfo)
{
	LOG4CPLUS_INFO(m_logger, pInfo);
}

void AlgorithmLog::DebugLog(const char* pInfo)
{
	LOG4CPLUS_DEBUG(m_logger, pInfo);
}
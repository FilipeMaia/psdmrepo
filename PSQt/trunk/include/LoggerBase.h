#ifndef LOGGERBASE_H
#define LOGGERBASE_H

//--------------------------
#include <map>
#include <iostream>    // std::cout
#include <fstream>     // std::ifstream(fname)
#include <sstream>     // stringstream
 
namespace PSQt {

//--------------------------
/**
 *  @ingroup PSQt LEVEL
 *
 *  @brief LEVEL - enumerator of message types/levels
 */ 

enum LEVEL {DEBUG=0, INFO, WARNING, CRITICAL, ERROR, NONE};

//--------------------------
/**
 *  @ingroup PSQt Record
 *
 *  @brief Record - struct for LoggerBase records
 */ 

struct Record {
  unsigned    number;
  double      time;
  std::string tstamp; 
  LEVEL       level;
  std::string name; 
  std::string msg; 

  std::string strRecordTotal();
  std::string strRecordBrief();
  std::string strRecord();
};

//--------------------------

std::string strLevel(const LEVEL& level=INFO) ;
LEVEL levelFromString(const std::string& str="INFO");
std::string strTimeStamp(const std::string& format=std::string("%Y-%m-%d-%H:%M:%S"));
double doubleTimeNow();

//--------------------------

// @addtogroup PSQt LoggerBase

/**
 *  @ingroup PSQt LoggerBase
 *
 *  @brief LoggerBase - base class for messaging system.
 *
 *  Accumulates messages as records in std::map<unsigned, Record> m_sslog;
 *  new_record(Record& rec) - callback for re-implementation in subclass.
 *  This class enherited by Logger.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see Logger, GUILogger, GUIMain
 *
 *  @version $Id$
 *
 *  @author Mikhail Dubrovin
 *
 *
 *
 *
 *
 *  @anchor interface
 *  @par<interface> Interface Description
 * 
 *  @li  Include
 *  @code
 *  #include "PSQt/LoggerBase.h"
 *  @endcode
 *
 *  @li Instatiation
 *  \n
 *  LoggerBase is inhereted and used by the class Logger
 */

//--------------------------

class LoggerBase
{
 public:

  /**
   *  @brief Base class for message logger
   *  
   *  @param[in] level - threshold level for collected messages. Messages with level lover than threshold are ignored.
   */ 
    LoggerBase(const LEVEL& level=INFO); 
    virtual ~LoggerBase(){}; 

    void message(const std::string& name=std::string(), const LEVEL& level=INFO, const std::string& msg=std::string());
    void message(const std::string& name=std::string(), const LEVEL& level=INFO, const std::stringstream& ss=std::stringstream());
    void print(const std::string& str=std::string());
    void setLevel(const LEVEL& level=INFO);
    LEVEL getLevel() { return m_level; }
    void saveLogInFile(const std::string& fname=std::string(), const bool& add_tstamp=true) ;
    std::string strRecordsForLevel(const LEVEL& thr=INFO);

 protected:
    virtual void new_record(Record& rec);

 private:

    unsigned    m_recnum;
    LEVEL       m_level;
    std::string m_start_tstamp;
    double      m_start_time;

    std::map<unsigned, Record> m_sslog;

    inline const char* _name_(){return "LoggerBase";}
    void startLoggerBase();
    std::string getFileName(const std::string& fname=std::string(), const bool& add_tstamp=true);
};

//--------------------------

//#define MsgInLog     LoggerBase::getLoggerBase()->message
//#define PrintMsg     LoggerBase::getLoggerBase()->print
//#define SetMsgLevel  Logger::getLogger()        ->setLevel
//#define SaveLog      LoggerBase::getLoggerBase()->saveLogInFile

//--------------------------

} // namespace PSQt

#endif // LOGGERBASE_H
//--------------------------
//--------------------------
//--------------------------


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
 *  @brief LEVEL - enumerator of message types
 */ 

enum LEVEL {DEBUG=0, INFO, WARNING, ERROR, CRITICAL};

//--------------------------
/**
 *  @ingroup PSQt Record
 *
 *  @brief Record - structure for LoggerBase
 */ 

struct Record {
  unsigned    number;
  double      time;
  std::string tstamp; 
  LEVEL       level;
  std::string name; 
  std::string msg; 

  std::string strRecordTotal();
  std::string strRecord();
};

//--------------------------

std::string strLevel(const LEVEL& level=INFO) ;
std::string strTimeStamp(const std::string& format=std::string("%Y-%m-%d-%H:%M:%S"));
double doubleTimeNow();

//--------------------------

// @addtogroup PSQt LoggerBase

/**
 *  @ingroup PSQt LoggerBase
 *
 *  @brief LoggerBase - base class for messaging system.
 *
 *  std::map<unsigned, Record> m_sslog; - accumulatior of records. 
 *  new_record(Record& rec) - callback for re-implementation in subclass.
 *
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see GUIMain
 *
 *  @version $Id:$
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
    LoggerBase(); 
    virtual ~LoggerBase(){}; 

    void message(const std::string& name=std::string(), const LEVEL& level=INFO, const std::string& msg=std::string());
    void message(const std::string& name=std::string(), const LEVEL& level=INFO, const std::stringstream& ss=std::stringstream());
    void print(const std::string& str=std::string());
    void setLevel(const LEVEL& level=INFO) { m_level=level; }
    LEVEL getLevel() { return m_level; }
    void saveLogInFile(const std::string& fname=std::string()) ;
    std::string strRecordsForLevel(const LEVEL& thr=INFO);

 protected:
    virtual void new_record(Record& rec);

 private:
    const std::string _name_(){return std::string("LoggerBase");}

    unsigned    m_recnum;
    LEVEL       m_level;
    std::string m_start_tstamp;
    double      m_start_time;

    std::map<unsigned, Record> m_sslog;

    void startLoggerBase();
    // Copy constructor and assignment are disabled by default
    //LoggerBase ( const LoggerBase& ) ;
    //LoggerBase& operator = ( const LoggerBase& ) ;
};

//--------------------------

//#define MsgInLog LoggerBase::getLoggerBase()->message
//#define PrintMsg LoggerBase::getLoggerBase()->print
//#define SaveLog  LoggerBase::getLoggerBase()->saveLogInFile

//--------------------------

} // namespace PSQt

#endif // LOGGERBASE_H
//--------------------------
//--------------------------
//--------------------------


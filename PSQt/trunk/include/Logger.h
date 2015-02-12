#ifndef LOGGER_H
#define LOGGER_H

//--------------------------
#include "PSQt/LoggerBase.h"
#include <QObject>
 
namespace PSQt {

//--------------------------

// @addtogroup PSQt Logger

/**
 *  @ingroup PSQt Logger
 *
 *  @brief Logger - singleton for base class LoggerBase - messaging system
 *
 *  Connects LoggerBase with GUILogger using method
 *  new_record(Record& rec) - callback for re-implementation in subclass,
 *  which emits signal with record for each new record above threshold level.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see LoggerBase, GUILogger, GUIMain
 *
 *  @version $Id$
 *
 *  @author Mikhail Dubrovin
 *
 *
 *  @anchor interface
 *  @par<interface> Interface Description
 * 
 *  @li  Include
 *  @code
 *  #include "PSQt/Logger.h"
 *  @endcode
 *
 *  @li Instatiation
 *  \n
 *  Instatiation is not requered, because this class is used as a singleton 
 *  with typedef-aliases to the methods like:
 *  @code
 *  Logger::getLogger()->some_method(...)
 *  @endcode
 *
 *
 *  @li Methods with aliases
 *  @code
 *  MsgInLog(_name_(), INFO, "some message is here"); // send message to the lagger
 *  Print("some message is here"); // just print, message is not saved in the logger
 *  SetMsgLevel(DEBUG); // change the level of messages for output and saving in file.
 *  @endcode
 *  @code
 *  SaveLog(); // save log in default file with name like: "log-2015-02-09-10:49:36.txt"
 *  // OR
 *  SaveLog("file-name.txt"); // save log in specified file
 *  @endcode
 *
 *
 *  @li Methods without aliases
 *  @code
 *  Logger::getLogger()->setLevel(DEBUG);
 *  LEVEL level = Logger::getLogger()->getLevel();
 *  std::string txt_error_msgs = Logger::getLogger()->strRecordsForLevel(ERROR);
 *  Logger::getLogger()-> ...
 *  @endcode
 *
 */

//--------------------------

class Logger : public QObject, public LoggerBase
{
 Q_OBJECT // macro is needed for connection of signals and slots

 public:
    static Logger* getLogger(const LEVEL& level=INFO); 
 
 protected:
    virtual void new_record(Record& rec);

 signals:
    void signal_new_record(Record&);

 private:
    static Logger* p_Logger;
 
    inline const char* _name_(){return "Logger";}
    Logger(const LEVEL& level=INFO); // private constructor! - singleton trick
    virtual ~Logger(){}; 
};

//--------------------------

#define MsgInLog     Logger::getLogger()->message
#define PrintMsg     Logger::getLogger()->print
#define SetMsgLevel  Logger::getLogger()->setLevel
#define SaveLog      Logger::getLogger()->saveLogInFile

//--------------------------

} // namespace PSQt

#endif // LOGGER_H
//--------------------------
//--------------------------
//--------------------------


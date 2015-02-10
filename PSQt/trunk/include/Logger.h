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
 *  @brief Logger - singleton for LoggerBase
 *
 *  Connects LoggerBase with GUILogger using method
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
 *  Print("some message is here"); // just print, message is not saved in the logger
 *  MsgInLog(_name_(), INFO, "some message is here"); // regular message to the lagger
 *  @endcode
 *  @code
 *  SaveLog(); // save log in default file with name like: "2015-02-09-10:49:36-log.txt"
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
    static Logger* getLogger(); 
 
 protected:
    virtual void new_record(Record& rec);

signals:
    void signal_new_record(Record&);

 private:
    static Logger* p_Logger;
 
    inline const std::string _name_(){return std::string("Logger");}

    Logger(); // private constructor! - singleton trick
    virtual ~Logger(){}; 
};

//--------------------------

#define MsgInLog  Logger::getLogger()->message
#define PrintMsg  Logger::getLogger()->print
#define SaveLog   Logger::getLogger()->saveLogInFile

//--------------------------

} // namespace PSQt

#endif // LOGGER_H
//--------------------------
//--------------------------
//--------------------------


#ifndef LOGGERNAMEWITHMPIRANK_H
#define LOGGERNAMEWITHMPIRANK_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class LoggerNameWithMpiRank
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------


/**
 * @ingroup Translator
 * @brief implements a name for logging that includes the mpi rank.
 *
 * example of use:
 * 
 * @code
 * LoggerNameWithMpiRank logger("myLogger");
 * 
 * ...
 *  
 * MsgLog(logger, info, "message");
 * @endcode
 *
 * If MPI_Init has been initialized before the logger is first used, 
 * all messages using the logger will have the name "myLogger mpiRnk=x"
 * where x is the rank of the process (in MPI_COMM_WORLD). Otherwise the
 * name will be "myLogger"
 *
 * @see MsgLog
 *
 */
class LoggerNameWithMpiRank {
 public:
  
  explicit LoggerNameWithMpiRank(const std::string & loggerName) 
    : m_loggerName(loggerName)
    , m_firstCall(true) {}
  
  operator const std::string &();

 protected:
  void setName();
  
 private:

  std::string m_loggerName;
  bool m_firstCall;

};

#endif

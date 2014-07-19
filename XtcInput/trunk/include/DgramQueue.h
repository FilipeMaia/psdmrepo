#ifndef XTCINPUT_DGRAMQUEUE_H
#define XTCINPUT_DGRAMQUEUE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DgramQueue.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/thread/condition.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/utility.hpp>
#include <queue>
#include <string>
#include <unistd.h>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "XtcInput/Dgram.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace XtcInput {

/// @addtogroup XtcInput

/**
 *  @ingroup XtcInput
 *
 *  @brief Synchronized datagram queue.
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class DgramQueue : boost::noncopyable {
public:

  typedef Dgram value_type ;

  // Default constructor
  DgramQueue (size_t maxSize) ;

  // Destructor
  ~DgramQueue () ;

  // add one more datagram to the queue, if the queue
  // is full already then wait until somebody calls pop()
  void push (const value_type& dg) ;

  // Producer thread may signal consumer thread that exception had
  // happened by calling push_exception() with non-empty message.
  void push_exception (const std::string& msg) ;

  // get one datagram from the head of the queue, if the queue is
  // empty then wait until somebody calls push(). If push_exception()
  // method was called then std::runtime_error exception will be thrown
  // wit the corresponding message.
  value_type pop() ;

  // get reference to datagram at the head of the queue, if the queue is
  // empty then wait until somebody calls push(). If push_exception()
  // method was called then std::runtime_error exception will be thrown
  // wit the corresponding message.
  value_type front();

  // completely erase all queue
  void clear() ;

protected:

private:

  // Data members
  size_t m_maxSize ;
  std::queue<value_type> m_queue ;
  std::string m_exception;
  boost::mutex m_mutex ;
  boost::condition m_condFull ;
  boost::condition m_condEmpty ;

};

} // namespace XtcInput

#endif // XTCINPUT_DGRAMQUEUE_H

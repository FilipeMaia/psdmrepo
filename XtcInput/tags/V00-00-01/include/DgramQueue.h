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
#include <unistd.h>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/xtc/Dgram.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace XtcInput {

/**
 *  Synchronized datagram queue.
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class DgramQueue : boost::noncopyable {
public:

  typedef Pds::Dgram* pointer_type ;

  // Default constructor
  DgramQueue ( size_t maxSize ) ;

  // Destructor
  ~DgramQueue () ;

  // add one more datagram to the queue, if the queue
  // is full already then wait until somebody calls pop()
  void push ( pointer_type dg ) ;

  // get one datagram from the head of the queue, if the queue is
  // empty then wait until somebody calls push()
  pointer_type pop() ;

protected:

private:

  // Data members
  size_t m_maxSize ;
  std::queue<pointer_type> m_queue ;
  boost::mutex m_mutex ;
  boost::condition m_condFull ;
  boost::condition m_condEmpty ;

};

} // namespace XtcInput

#endif // XTCINPUT_DGRAMQUEUE_H

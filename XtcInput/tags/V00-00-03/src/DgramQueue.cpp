//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DgramQueue...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XtcInput/DgramQueue.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace XtcInput {

//----------------
// Constructors --
//----------------
DgramQueue::DgramQueue ( size_t maxSize )
  : m_maxSize ( maxSize )
  , m_queue()
  , m_mutex()
  , m_condFull()
  , m_condEmpty()
{
}

//--------------
// Destructor --
//--------------
DgramQueue::~DgramQueue ()
{
}

// add one more datagram to the queue, if the queue
// is full already then wait until somebody calls pop()
void
DgramQueue::push ( DgramQueue::pointer_type dg )
{
  boost::mutex::scoped_lock qlock ( m_mutex ) ;

  // wait unil we have an empty slot
  while ( m_queue.size() >= m_maxSize ) {
    m_condFull.wait( qlock ) ;
  }

  // store the packet
  m_queue.push ( dg ) ;

  // tell anybody waiting for queue to become non-empty
  m_condEmpty.notify_one () ;

}

// get one datagram from the head of the queue, if the queue is
// empty then wait until somebody calls push()
DgramQueue::pointer_type
DgramQueue::pop()
{
  boost::mutex::scoped_lock qlock ( m_mutex ) ;

  // wait unil we have something in the queue
  while ( m_queue.empty() ) {
    m_condEmpty.wait( qlock ) ;
  }

  // get a packet
  pointer_type p = m_queue.front() ;
  m_queue.pop() ;

  // tell anybody waiting for queue to become non-full
  m_condFull.notify_one () ;

  return p ;
}

} // namespace XtcInput

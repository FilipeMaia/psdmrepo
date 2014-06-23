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
  , m_exception()
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
DgramQueue::push (const value_type& dg)
{
  boost::mutex::scoped_lock qlock ( m_mutex ) ;

  // wait until we have an empty slot
  while ( m_queue.size() >= m_maxSize ) {
    m_condFull.wait( qlock ) ;
  }

  // store the packet
  m_queue.push ( dg ) ;

  // tell anybody waiting for queue to become non-empty
  m_condEmpty.notify_one () ;

}

// Producer thread may signal consumer thread that exception had
// happened by calling push_exception() with non-empty message.
void
DgramQueue::push_exception (const std::string& msg)
{
  boost::mutex::scoped_lock qlock(m_mutex);

  // store the message
  m_exception = msg;

  // tell anybody waiting for new data
  m_condEmpty.notify_one();
}

// get one datagram from the head of the queue, if the queue is
// empty then wait until somebody calls push()
DgramQueue::value_type
DgramQueue::pop()
{
  boost::mutex::scoped_lock qlock ( m_mutex ) ;

  // wait until we have something in the queue
  while (m_exception.empty() and m_queue.empty()) {

    m_condEmpty.wait( qlock ) ;

    // throw exception if non-empty, reset exception message
    if (not m_exception.empty()) {
      std::string msg;
      msg.swap(m_exception);
      throw std::runtime_error(msg);
    }

    if (not m_queue.empty()) break;
  }

  // get a packet
  value_type p = m_queue.front() ;
  m_queue.pop() ;

  // tell anybody waiting for queue to become non-full
  m_condFull.notify_one () ;

  return p ;
}

// get reference to datagram at the head of the queue, if the queue is
// empty then wait until somebody calls push()
DgramQueue::value_type  
DgramQueue::front() {

  boost::mutex::scoped_lock qlock ( m_mutex ) ;

  // wait until we have something in the queue
  while (m_exception.empty() and m_queue.empty()) {

    m_condEmpty.wait( qlock ) ;

    // throw exception if non-empty, reset exception message
    if (not m_exception.empty()) {
      std::string msg;
      msg.swap(m_exception);
      throw std::runtime_error(msg);
    }

    if (not m_queue.empty()) break;
  }

  return m_queue.front();
}

// completely erase all queue
void 
DgramQueue::clear()
{
  boost::mutex::scoped_lock qlock ( m_mutex ) ;

  // erase everything
  while (not m_queue.empty()) m_queue.pop() ;

  // tell anybody waiting for queue to become non-full
  m_condFull.notify_one () ;
}

} // namespace XtcInput

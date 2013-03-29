//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcIterator...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XtcInput/XtcIterator.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "XtcInput/Exceptions.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {
  
  const char* logger = "XtcInput.XtcIterator";
  
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace XtcInput {

//----------------
// Constructors --
//----------------
XtcIterator::XtcIterator ( Pds::Xtc* xtc )
  : m_initial(xtc)
  , m_xtc()
  , m_off() 
{
}

//--------------
// Destructor --
//--------------
XtcIterator::~XtcIterator ()
{
}

Pds::Xtc* 
XtcIterator::next()
{
  // first call
  if (Pds::Xtc* xtc = m_initial) {
    MsgLog(logger, debug, "XtcIterator: first call");
    if ( xtc->contains.id() == Pds::TypeId::Id_Xtc ) {
      // if a container then store in a stack
      MsgLog(logger, debug, "XtcIterator: push " << m_off.size());
      m_xtc.push(xtc);
      m_off.push(0);
    }
    m_initial = 0;
    return xtc;
  }
  
  // if nothing left return 0
  if (m_xtc.empty()) {
    MsgLog(logger, debug, "XtcIterator: finished");
    return 0;
  }
  
  // finished iterating this xtc, drop it, move to previous
  while (m_off.top() >= m_xtc.top()->sizeofPayload()) {
    m_off.pop();
    m_xtc.pop();
    MsgLog(logger, debug, "XtcIterator: pop " << m_off.size());
    // no more XTCs left, stop
    if(m_off.empty()) {
      MsgLog(logger, debug, "XtcIterator: finished");
      return 0;
    }
  }  

  MsgLog(logger, debug, "XtcIterator: payload size = " << m_xtc.top()->sizeofPayload());
  MsgLog(logger, debug, "XtcIterator: offset size = " << m_off.top());

  // at this point the XTC in stack can only be of Pds::TypeId::Id_Xtc type
  Pds::Xtc* next = (Pds::Xtc*)(m_xtc.top()->payload() + m_off.top());
  MsgLog(logger, debug, "XtcIterator: next payload = " << next->sizeofPayload());
  MsgLog(logger, debug, "XtcIterator: next type = " << Pds::TypeId::name(next->contains.id()));

  // adjust remaining size
  int nextoff = m_off.top() + next->sizeofPayload() + sizeof(Pds::Xtc);
  MsgLog(logger, debug, "XtcIterator: next offset = " << nextoff);
  if ( nextoff >  m_xtc.top()->sizeofPayload()) {
    // looks badly corrupted, print a warning
    MsgLog(logger, warning, "Corrupted XTC, size out of range, xtc payload size: "
           << m_xtc.top()->sizeofPayload()
           << ", contained data size: " << nextoff);
  }
  MsgLog(logger, debug, "XtcIterator: set offset " << nextoff);
  m_off.top() = nextoff;

  if ( next->contains.id() == Pds::TypeId::Id_Xtc ) {
    // add it to stack too
    MsgLog(logger, debug, "XtcIterator: push " << m_off.size());
    m_xtc.push(next);
    m_off.push(0);
  }
  
  return next;
}

} // namespace XtcInput

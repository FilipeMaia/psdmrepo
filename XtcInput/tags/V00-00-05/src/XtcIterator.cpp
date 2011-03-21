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
  
  const char* logger = "XtcIterator";
  
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
    if ( xtc->contains.id() == Pds::TypeId::Id_Xtc ) {
      // if a container then store in a stack
      m_xtc.push(xtc);
      m_off.push(0);
    }
    m_initial = 0;
    return xtc;
  }
  
  // if nothing left return 0
  if (m_xtc.empty()) {
    return 0;
  }
  
  // finished iterating this xtc, drop it, move to previous
  while (m_off.top() == m_xtc.top()->sizeofPayload()) {
    m_off.pop();
    m_xtc.pop();
    // no more XTCs left, stop
    if(m_off.empty()) {
      return 0;
    }
  }  

  // at this point the XTC in stack can only be of Pds::TypeId::Id_Xtc type
  Pds::Xtc* next = (Pds::Xtc*)(m_xtc.top()->payload() + m_off.top());

  // adjust remaining size
  int nextoff = m_off.top() + next->sizeofPayload() + sizeof(Pds::Xtc);
  if ( nextoff >  m_xtc.top()->sizeofPayload()) {
    // looks badly corrupted, throw exception
    throw XTCGenException ("Corrupted XTC, size out of range");
  }
  m_off.top() = nextoff;

  if ( next->contains.id() == Pds::TypeId::Id_Xtc ) {
    // add it to stack too
    m_xtc.push(next);
    m_off.push(0);
  }
  
  return next;
}

} // namespace XtcInput

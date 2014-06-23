//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Dgram...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XtcInput/Dgram.h"

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

/**
 *  @brief This method will be used in place of regular delete.
 */
void 
Dgram::destroy(const Pds::Dgram* dg) 
{ 
  delete [] (const char*)dg; 
}

/**
 *  @brief Factory method which wraps existing object into a smart pointer.
 */
Dgram::ptr 
Dgram::make_ptr(Pds::Dgram* dg)
{
  return ptr(dg, &Dgram::destroy);
}


/**
 *  @brief Factory method which copies existing datagram and wraps new 
 *  object into a smart pointer.
 */
Dgram::ptr 
Dgram::copy(Pds::Dgram* dg) 
{
  // make a copy
  char* dgbuf = (char*)dg ;
  size_t dgsize = sizeof(Pds::Dgram) + dg->xtc.sizeofPayload();
  char* buf = new char[dgsize] ;
  std::copy( dgbuf, dgbuf+dgsize, buf ) ;
  return ptr((Pds::Dgram*)buf, &destroy);
}

bool Dgram::operator< (const Dgram& other) const {
  // a workaround for the fact that pdsdata clocktime doesn't
  // implement operator<.
  if (m_dg->seq.clock() > other.m_dg->seq.clock()) return 0;
  if (m_dg->seq.clock() == other.m_dg->seq.clock()) return 0;
  return 1;
}

} // namespace XtcInput

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
#include <sstream>

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
  // empty dgrams are always last
  if (empty()) return false;
  if (other.empty()) return true;
  
  // a workaround for the fact that pdsdata clocktime doesn't
  // implement operator<.
  if (m_dg->seq.clock() > other.m_dg->seq.clock()) return 0;
  if (m_dg->seq.clock() == other.m_dg->seq.clock()) return 0;
  return 1;
}

std::string Dgram::dumpStr(const XtcInput::Dgram &dg) {
  if (dg.empty()) return "empty dgram";
  std::ostringstream msg;
  const Pds::Dgram *dgram = dg.dg().get();
  const Pds::Sequence & seq = dgram->seq;
  const Pds::Env & env = dgram->env;
  const Pds::ClockTime & clock = seq.clock();
  const Pds::TimeStamp & stamp = seq.stamp();
  msg << "tp=" << int(seq.type())
      << " sv=" << Pds::TransitionId::name(seq.service())
      << " ex=" << seq.isExtended()
      << " ev=" << seq.isEvent()
      << " sec=" << std::hex << clock.seconds()
      << " nano=" << std::hex << clock.nanoseconds()
      << " tcks=" << std::hex << stamp.ticks()
      << " fid=" << stamp.fiducials()
      << " ctrl=" << stamp.control()
      << " vec=" << stamp.vector()
      << " env=" << env.value()
      << " file=" << dg.file().path();
  return msg.str();
}
  
} // namespace XtcInput

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2OXtcSrc...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/O2OXtcSrc.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/xtc/BldInfo.hh"
#include "pdsdata/xtc/DetInfo.hh"
#include "pdsdata/xtc/ProcInfo.hh"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // convert ProcInfo to name
  std::string toName( const Pds::ProcInfo& info )
  {
    std::ostringstream str ;
    uint32_t ip = info.ipAddr() ;
    str << info.processId() << '@' << ((ip>>24)&0xff) << '.'
        << ((ip>>16)&0xff) << '.' << ((ip>>8)&0xff) << '.' << (ip&0xff) ;
    return str.str() ;
  }

  // convert BldInfo to name
  std::string toName( const Pds::BldInfo& info )
  {
    return Pds::BldInfo::name(info) ;
  }

  // convert DetInfo to name
  std::string toName( const Pds::DetInfo& info )
  {
    std::ostringstream str ;
    str << Pds::DetInfo::name(info.detector()) << '.' << info.detId()
        << ':' << Pds::DetInfo::name(info.device()) << '.' << info.devId() ;
    return str.str() ;
  }

  // convert Src to name
  std::string toName( const Pds::Src& src )
  {
    if ( src.level() == Pds::Level::Source ) {
      const Pds::DetInfo& info = static_cast<const Pds::DetInfo&>( src ) ;
      return toName( info ) ;
    } else if ( src.level() == Pds::Level::Reporter ) {
      const Pds::BldInfo& info = static_cast<const Pds::BldInfo&>( src ) ;
      return toName( info ) ;
    } else if ( src.level() == Pds::Level::Control ) {
      // special case for control data
      return "Control";
    } else {
      const Pds::ProcInfo& info = static_cast<const Pds::ProcInfo&>( src ) ;
      return toName( info ) ;
    }
  }


}


//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

// get the name of the source
std::string
O2OXtcSrc::name() const
{
  if ( m_src.empty() ) return std::string() ;

  const Pds::Src& top = m_src.back() ;
  return ::toName(top) ;
}

} // namespace O2OTranslator

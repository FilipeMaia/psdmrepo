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
#include "Lusi/Lusi.h"

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
    if ( src.level() == Pds::Level::Segment ) {
      const Pds::ProcInfo& info = static_cast<const Pds::ProcInfo&>( src ) ;
      return toName( info ) ;
    } else {
      const Pds::DetInfo& info = static_cast<const Pds::DetInfo&>( src ) ;
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
  if ( top.level() == Pds::Level::Control ) {

    // at Control level we need to use the host name from upper Segment level
    for ( unsigned i = m_src.size()-1 ; i > 0 ; -- i ) {
      const Pds::Src& src = m_src[i-1] ;
      if ( src.level() == Pds::Level::Segment ) {
        return ::toName(src) ;
      }
    }

    // if segment was not found then use control as source
    return ::toName(top) ;

  } else {

    // in all other cases format topmost
    return ::toName(top) ;

  }

}

} // namespace O2OTranslator

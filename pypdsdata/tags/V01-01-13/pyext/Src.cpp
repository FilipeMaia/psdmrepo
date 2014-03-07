//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Src...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "Src.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "BldInfo.h"
#include "DetInfo.h"
#include "ProcInfo.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace pypdsdata {

/**
 *  Method that knows how to print the content of Src.
 */
void 
Src::print(std::ostream& out, const Pds::Src& src)
{
  if ( src.level() == Pds::Level::Reporter ) {
    
    const Pds::BldInfo& info = static_cast<const Pds::BldInfo&>(src);
    if (info.type() < Pds::BldInfo::NumberOf) {
      out << "BldInfo(" << Pds::BldInfo::name(info) << ")";
    } else {
      out << "BldInfo(" << int(info.type()) << ")";
    }
    
  } else if ( src.level() == Pds::Level::Source ) {
    
    const Pds::DetInfo& info = static_cast<const Pds::DetInfo&>(src);
    out << "DetInfo(" << Pds::DetInfo::name(info.detector()) << "-" << info.detId() 
        << "|" << Pds::DetInfo::name(info.device()) << "-" << info.devId() << ")";

  } else {
    
    const Pds::ProcInfo& info = static_cast<const Pds::ProcInfo&>(src);
    unsigned ip = info.ipAddr() ;
    out << "ProcInfo(" << Pds::Level::name(info.level()) << ", " << info.processId()  << ", " 
        << ((ip>>24)&0xff) << '.' << ((ip>>16)&0xff) << '.' << ((ip>>8)&0xff)  << '.' << (ip&0xff) << ")";

  }

}

/// convert src to python object
PyObject* 
Src::PyObject_FromPds(const Pds::Src& src)
{
  if ( src.level() == Pds::Level::Reporter ) {
    const Pds::BldInfo& info = static_cast<const Pds::BldInfo&>(src);
    return pypdsdata::BldInfo::PyObject_FromPds(info);
  } else if ( src.level() == Pds::Level::Source ) {
    const Pds::DetInfo& info = static_cast<const Pds::DetInfo&>(src);
    return pypdsdata::DetInfo::PyObject_FromPds(info);
  } else {
    const Pds::ProcInfo& info = static_cast<const Pds::ProcInfo&>(src);
    return pypdsdata::ProcInfo::PyObject_FromPds(info);
  }
}

} // namespace pypdsdata

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EventKey...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSEvt/EventKey.h"
#include "PSEvt/TypeInfoUtils.h"
 
//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>
#include <cxxabi.h>
#include <stdlib.h>

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


  void printDetInfo(std::ostream& str, const Pds::DetInfo& src)
  {
    str << "DetInfo(";
    if (src.detector() == 0xff) {
      str << '*';
    } else {
      str << Pds::DetInfo::name(src.detector());
    }
    str << '.';
    if (src.detId() == 0xff) {
      str << '*';
    } else {
      str << src.detId();
    }
    str << ':';
    if (src.device() == 0xff) {
      str << '*';
    } else {
      str << Pds::DetInfo::name(src.device());
    }
    str << '.';
    if (src.devId() == 0xff) {
      str << '*';
    } else {
      str << src.devId();
    }
    str << ')';
  }

  void printBldInfo(std::ostream& str, const Pds::BldInfo& src)
  {
    str << "BldInfo(";
    if (unsigned(src.type()) != 0xffffffff) str << Pds::BldInfo::name(src);
    str << ')';
  }

  void printProcInfo(std::ostream& str, const Pds::ProcInfo& src)
  {
    uint32_t ip = src.ipAddr();
    str << "ProcInfo(" 
        << ((ip>>24)&0xff) 
        << '.' << ((ip>>16)&0xff)
        << '.' << ((ip>>8)&0xff) 
        << '.' << (ip&0xff) 
        << ", pid=" << src.processId() 
        << ')';
  }

  void print(std::ostream& str, const Pds::Src& src)
  {
    if (src.level() == Pds::Level::Source) {
      printDetInfo(str, static_cast<const Pds::DetInfo&>(src));
    } else if (src.level() == Pds::Level::Reporter) {
      printBldInfo(str, static_cast<const Pds::BldInfo&>(src));
    } else if (src.level() == Pds::Level::NumberOfLevels) {
      // special match-anything source, empty string
    } else {
      printProcInfo(str, static_cast<const Pds::ProcInfo&>(src));
    }
  }
  
  void print(std::ostream& str, const std::type_info* typeinfo)
  {
    str << PSEvt::TypeInfoUtils::typeInfoRealName(typeinfo);
  }

}


//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSEvt {

bool 
EventKey::operator<(const EventKey& other) const 
{
  int src_diff = PSEvt::cmpPdsSrc(this->m_src, other.m_src);
  if (src_diff<0) return true;
  if (src_diff>0) return false;
  if( TypeInfoUtils::lessTypeInfoPtr()(m_typeinfo, other.m_typeinfo) ) return true;
  if( TypeInfoUtils::lessTypeInfoPtr()(other.m_typeinfo, m_typeinfo) ) return false;
  if (m_key < other.m_key) return true;
  return false;
}


// format the key
void 
EventKey::print(std::ostream& str) const
{
  str << "EventKey(type=";
  ::print(str, m_typeinfo);
  if (m_src == anySource()) {
    str << ", src=AnySource";
  } else if (validSrc()) {
    str << ", src=";
    ::print(str, m_src);
  }
  if (not m_key.empty()) {
    str << ", key=" << m_key; 
  }
  str << ')';
}

// Compare two Src objects, ignores process ID.
int cmpPdsSrc(const Pds::Src& lhs, const Pds::Src& rhs)
{
  // ignore PID in comparison
  int diff = int(lhs.level()) - int(rhs.level());
  if (diff != 0) return diff;
  return int(lhs.phy()) - int(rhs.phy());
}


} // namespace PSEvt

namespace Pds {
// Helper method to format Pds::Src to a standard stream
std::ostream&
operator<<(std::ostream& out, const Pds::Src& src)
{
  ::print(out, src);
  return out;
}
}

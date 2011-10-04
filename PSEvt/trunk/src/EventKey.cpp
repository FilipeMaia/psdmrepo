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

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>
#include <cxxabi.h>

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

  // compare two Src objects
  int cmp(const Pds::Src& lhs, const Pds::Src& rhs)
  {
    // ignore PID in comparison
    int diff = int(lhs.level()) - int(rhs.level());
    if (diff != 0) return diff;
    return int(lhs.phy()) - int(rhs.phy());
  }

  void printDetInfo(std::ostream& str, const Pds::DetInfo& src)
  {
    str << "DetInfo("
        << Pds::DetInfo::name(src.detector()) 
        << '.' << src.detId() 
        << ':' << Pds::DetInfo::name(src.device()) 
        << '.' << src.devId() 
        << ')';
  }

  void printBldInfo(std::ostream& str, const Pds::BldInfo& src)
  {
    str << "BldInfo(" << Pds::BldInfo::name(src) << ')';
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
    } else {
      printProcInfo(str, static_cast<const Pds::ProcInfo&>(src));
    }
  }
  
  void print(std::ostream& str, const std::type_info* typeinfo)
  {
    int status;
    char* realname = abi::__cxa_demangle(typeinfo->name(), 0, 0, &status);
    str << realname;
    free(realname);
  }

}


//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSEvt {

bool 
EventKey::operator<(const EventKey& other) const 
{
  int src_diff = ::cmp(this->m_src, other.m_src);
  if (src_diff<0) return true;
  if (src_diff>0) return false;
  if( m_typeinfo->before(*other.m_typeinfo) ) return true;
  if( other.m_typeinfo->before(*m_typeinfo) ) return false;
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

} // namespace PSEvt

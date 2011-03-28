//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Source...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSEvt/Source.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <vector>
#include <cassert>
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/lexical_cast.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "PSEvt/Exceptions.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {
  
  const char* logger = "PSEvt::Source";

  // Parse spec tring and return Src or throw
  Pds::Src parse(std::string spec);
  
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSEvt {

/**
 *  @brief Match for any source.
 *  
 *  This object will match any source.
 */
Source::Source () 
  : m_src(Pds::Level::NumberOfLevels) 
{
}

Source::Source (const Pds::Src& src) 
  : m_src(src) 
{
} 

/**
 *  @brief Exact match for DetInfo source.
 *  
 *  This object will match fully-specified DetInfo source.
 */
Source::Source(Pds::DetInfo::Detector det, uint32_t detId, Pds::DetInfo::Device dev, uint32_t devId) 
  : m_src(Pds::DetInfo(0, det, detId, dev, devId))
{
}

/**
 *  @brief Exact match for BldInfo.
 *  
 *  This object will match fully-specified BldInfo source.
 */
Source::Source(Pds::BldInfo::Type type)
  : m_src(Pds::BldInfo(0, type))
{
}

Source::Source (const std::string& spec)
  : m_src(::parse(spec))
{
  
}

Source& 
Source::operator=(const std::string& spec)
{
  Source src(spec);
  m_src = src.m_src;
  return *this;
}

/**
 *  @brief Match source with Pds::Src object.
 */
bool 
Source::match(const Pds::Src& src) const
{
  if (m_src == Pds::Src()) {
    
    // no-source must match exactly
    return src == Pds::Src();
    
  } else if (m_src.level() == Pds::Level::NumberOfLevels) {
    
    // any-match matches all
    return true;

  } else if (int(src.level()) > Pds::Level::NumberOfLevels) {
    
    // strange address (probably no-source)
    return false;
    
  } else if (m_src.level() == Pds::Level::Source) {
    
    // DetInfo match, source must be the same level
    if (src.level() != Pds::Level::Source) return false;

    const Pds::DetInfo& minfo = static_cast<const Pds::DetInfo&>(m_src);
    const Pds::DetInfo& info = static_cast<const Pds::DetInfo&>(src);
    
    if (int(minfo.detector()) != 255 and minfo.detector() != info.detector()) return false;
    if (int(minfo.device()) != 255 and minfo.device() != info.device()) return false;
    if (minfo.detId() != 255 and minfo.detId() != info.detId()) return false;
    if (minfo.devId() != 255 and minfo.devId() != info.devId()) return false;
    return true;
    
  } else if (m_src.level() == Pds::Level::Reporter) {

    // BldInfo match, source must be the same level
    if (src.level() != Pds::Level::Reporter) return false;

    const Pds::BldInfo& minfo = static_cast<const Pds::BldInfo&>(m_src);
    const Pds::BldInfo& info = static_cast<const Pds::BldInfo&>(src);

    if (uint32_t(minfo.type()) != 0xffffffff and minfo.type() != info.type()) return false;
    return true;
    
  } else {
    
    // ProcInfo match, level can be anything except Source and Reporter
    if (src.level() == Pds::Level::Source) return false;
    if (src.level() == Pds::Level::Reporter) return false;
    
    const Pds::ProcInfo& minfo = static_cast<const Pds::ProcInfo&>(m_src);
    const Pds::ProcInfo& info = static_cast<const Pds::ProcInfo&>(src);

    if (minfo.ipAddr() != 0xffffffff and minfo.ipAddr() != info.ipAddr()) return false;
    return true;

  }
  
}

/// Returns true if it is exact match
bool 
Source::isExact() const
{
  if (m_src.level() == Pds::Level::NumberOfLevels) {
    // match-any object is not exact
    return false;
  }
  
  if (m_src == Pds::Src()) {
    // no-source match is exact
    return true;
  }

  if (m_src.level() == Pds::Level::Source) {
    
    const Pds::DetInfo& info = static_cast<const Pds::DetInfo&>(m_src);
    return info.detector() != 255 and info.device() != 255 and
        info.detId() != 255 and info.devId() != 255;
    
  } else if (m_src.level() == Pds::Level::Reporter) {
    
    const Pds::BldInfo& info = static_cast<const Pds::BldInfo&>(m_src);
    return uint32_t(info.type()) != 0xffffffff;
  
  } else {
    
    const Pds::ProcInfo& info = static_cast<const Pds::ProcInfo&>(m_src);
    return info.ipAddr() != 0xffffffff;

  }

}

} // namespace PSEvt


namespace {

  
// Strip type and parentheses from "Type(...)"
std::string
stripType(const std::string& spec, int typeLen)
{
  std::string sspec(spec, typeLen);
  boost::algorithm::trim(sspec);

  if (sspec.empty()) throw PSEvt::ExceptionSourceFormat(ERR_LOC, spec);
  if (sspec[0] != '(') throw PSEvt::ExceptionSourceFormat(ERR_LOC, spec);
  
  std::string::size_type p2 = sspec.rfind(')');
  if (p2 == std::string::npos) {
    throw PSEvt::ExceptionSourceFormat(ERR_LOC, spec);
  }

  // remove everything after and including (
  sspec.erase(p2);
  // remove leading (
  sspec.erase(0, 1);

  // strip leading/trailing blanks too
  boost::algorithm::trim(sspec);
  return sspec;
}

// Parse BldInfo data, may be empty string or a string returned
// from Pds::BldInfo::name()
bool parseBldInfo(std::string spec, Pds::Src *src)
{
  MsgLog(logger, debug, "Source::parseBldInfo: spec = '" << spec << "'");

  if (spec.empty()) {
    *src = Pds::BldInfo(0, Pds::BldInfo::Type(0xffffffff));
    return true;
  }

  // try all known BldInfo names
  for (int i = 0; i < Pds::BldInfo::NumberOf; ++ i) {
    Pds::BldInfo info(0, Pds::BldInfo::Type(i));
    if (spec == Pds::BldInfo::name(info)) {
      *src = info;
      return true;
    }
  }
  
  return false;
}

// Converst string '[Name][.Num]' into pair Name+Number
// Throws boost::bad_lexical_cast if number is not formatted properly
std::pair<std::string, unsigned> splitDetId(const std::string& spec, char sep)
{
  std::pair<std::string, unsigned> res(std::string(), 255);
  std::string::size_type p1 = spec.find(sep);
  if(p1 == std::string::npos) {
    res.first = spec;
  } else {
    res.first = std::string(spec, 0, p1);
    std::string idStr = std::string(spec, p1+1);
    if (not (idStr.empty() or idStr == "*")) {
      res.second = boost::lexical_cast<unsigned>(idStr);
    }
  }
  if (res.first == "*") res.first.clear();
  return res;
}

// Parse DetInfo string, may be empty string or a string returned
// from Pds::BldInfo::name()
bool parseDetInfo(const std::string& spec, Pds::Src *src, char sep1, char sep2)
{
  // string cannot be empty here
  assert(not spec.empty());
  
  std::string detSpec;
  std::string devSpec;
  std::string::size_type p1 = spec.find(sep1);
  if(p1 == std::string::npos) {
    detSpec = spec;
  } else {
    detSpec = std::string(spec, 0, p1);
    devSpec = std::string(spec, p1+1);
  }

  std::pair<std::string, unsigned> detPair;
  std::pair<std::string, unsigned> devPair;
  try {
    detPair = splitDetId(detSpec, sep2);
    devPair = splitDetId(devSpec, sep2);
  } catch (const boost::bad_lexical_cast& ex) {
    return false;
  }
  
  Pds::DetInfo::Detector det = Pds::DetInfo::Detector(255);
  Pds::DetInfo::Device dev = Pds::DetInfo::Device(255);
  if (not detPair.first.empty()) {
    for(int i = 0; i < Pds::DetInfo::NumDetector; ++ i) {
      if (detPair.first == Pds::DetInfo::name(Pds::DetInfo::Detector(i))) {
        det = Pds::DetInfo::Detector(i);
        break;
      }
    }
    if (det == 255) return false;
  }
  if (not devPair.first.empty()) {
    for(int i = 0; i < Pds::DetInfo::NumDevice; ++ i) {
      if (devPair.first == Pds::DetInfo::name(Pds::DetInfo::Device(i))) {
        dev = Pds::DetInfo::Device(i);
        break;
      }
    }
    if (dev == 255) return false;
  }

  *src = Pds::DetInfo(0, det, detPair.second, dev, devPair.second);
  return true;
}

// Parse DetInfo string, may be empty string or a string returned
// from Pds::BldInfo::name()
bool parseDetInfo(std::string spec, Pds::Src *src)
{
  MsgLog(logger, debug, "Source::parseDetInfo: spec = '" << spec << "'");

  if (spec.empty()) {
    *src = Pds::DetInfo(0, Pds::DetInfo::Detector(0xff), 0xff, Pds::DetInfo::Device(0xff), 0xff);
    return true;
  }
  
  if (parseDetInfo(spec, src, ':', '.')) return true;
  if (parseDetInfo(spec, src, '|', '-')) return true;  
  return false;
}


// Parse ProcInfo string, may be empty string or "NNN.NNN.NNN.NNN"
bool parseProcInfo(std::string spec, Pds::Src *src)
{
  MsgLog(logger, debug, "Source::parseProcInfo: spec = '" << spec << "'");

  if (spec.empty()) {
    // level can be anything except Source or Reporter
    *src = Pds::ProcInfo(Pds::Level::Segment, 0, 0xffffffff);
    return true;
  }

  std::vector<std::string> octets;
  boost::algorithm::split(octets, spec, boost::is_any_of("."));
  if (octets.size() != 4) return false;

  uint32_t ip;
  try {
    
    unsigned oct0 = boost::lexical_cast<unsigned>(octets[0]);
    unsigned oct1 = boost::lexical_cast<unsigned>(octets[1]);
    unsigned oct2 = boost::lexical_cast<unsigned>(octets[2]);
    unsigned oct3 = boost::lexical_cast<unsigned>(octets[3]);
    if (oct0 > 255 or oct1 > 255 or oct2 > 255 or oct3 > 255) return false;
    ip = (oct0 << 24) | (oct1 << 16) | (oct2 << 8) | oct3;
    
  } catch (const boost::bad_lexical_cast& ex) {
    return false;
  }
  
  // level can be anything except Source or Reporter
  *src = Pds::ProcInfo(Pds::Level::Segment, 0, ip);
  return true;  
}

  
// Parse spec string and return Src or throw
Pds::Src 
parse(std::string spec)
{
  // strip leading/trailing blanks
  boost::algorithm::trim(spec);

  MsgLog(logger, debug, "Source::parse: spec = '" << spec << "'");

  // empty string means match anything
  if (spec.empty()) return Pds::Src(Pds::Level::NumberOfLevels);

  if (boost::algorithm::starts_with(spec, "DetInfo")) {
    
    Pds::Src src;
    if (not parseDetInfo(stripType(spec, 7), &src)) {
      throw PSEvt::ExceptionSourceFormat(ERR_LOC, spec);
    }
    return src;

  } else if (boost::algorithm::starts_with(spec, "BldInfo")) {

    Pds::Src src;
    if (not parseBldInfo(stripType(spec, 7), &src)) { 
      throw PSEvt::ExceptionSourceFormat(ERR_LOC, spec);
    }
    return src;
    
  } else if (boost::algorithm::starts_with(spec, "ProcInfo")) {

    Pds::Src src;
    if (not parseProcInfo(stripType(spec, 8), &src)) {
      throw PSEvt::ExceptionSourceFormat(ERR_LOC, spec);
    }
    return src;
    
  } else {
  
    Pds::Src src;
    if (parseBldInfo(spec, &src)) return src;
    if (parseDetInfo(spec, &src)) return src;
    throw PSEvt::ExceptionSourceFormat(ERR_LOC, spec);
    
  }
}
  
}

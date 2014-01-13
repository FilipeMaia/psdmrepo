#ifndef PSEVT_SOURCE_H
#define PSEVT_SOURCE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Source.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <iosfwd>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSEvt/AliasMap.h"
#include "pdsdata/xtc/BldInfo.hh"
#include "pdsdata/xtc/DetInfo.hh"
#include "pdsdata/xtc/ProcInfo.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSEvt {

/**
 *  @ingroup PSEvt
 *  
 *  @brief This class implements source matching for finding data 
 *  inside event.
 *  
 *  Event dictionary has to support location of the event data without 
 *  complete source address specification. This class provides facility
 *  for matching the data source address against partially-specified 
 *  match.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see Event
 *
 *  @version \$Id$
 *
 *  @author Andrei Salnikov
 */

class Source {
public:

  /**
   *  Helper class which provides logic for matching Source
   *  values to Src instances.
   */
  class SrcMatch {
  public:

    SrcMatch(const Pds::Src& src)
        : m_src(src)
    {
    }

    /// Match source with Pds::Src object.
    bool match(const Pds::Src& src) const;

    /// Returns true if matches no-source only
    bool isNoSource() const
    {
      return m_src == Pds::Src();
    }

    /// Returns true if it is exact match, no-source is also exact.
    bool isExact() const;

    /// Returns internal Src representation
    const Pds::Src& src() const
    {
      return m_src;
    }

  private:
    Pds::Src m_src;
  };

  /// Special enum type to signify objects without source
  enum NoSource {
    null ///< Special constant to be used as argument for constructor
  };

  /**
   *  @brief Make source which matches objects without source only.
   */
  explicit Source(NoSource)
      : m_src()
  {
  }

  /**
   *  @brief Make source which matches any source.
   *  
   *  This object will match any source.
   */
  Source();

  /**
   *  @brief Exact match for source.
   *  
   *  This object will match fully-specified source. Note that Source(Pds::Src())
   *  is equivalent to Source(null).
   */
  explicit Source(const Pds::Src& src);

  /**
   *  @brief Exact match for DetInfo source.
   *  
   *  This object will match fully-specified DetInfo source.
   */
  Source(Pds::DetInfo::Detector det, uint32_t detId, Pds::DetInfo::Device dev, uint32_t devId);

  /**
   *  @brief Exact match for BldInfo.
   *  
   *  This object will match fully-specified BldInfo source.
   */
  explicit Source(Pds::BldInfo::Type type);

  /**
   *  @brief Matching specified via string.
   *  
   *  Argument string can be either alias name or match string in one of these formats:
   *    "" - match anything
   *    "DetInfo(det.detId:dev.devId)" - fully or partially specified DetInfo
   *    "det.detId:dev.devId" - same as above
   *    "DetInfo(det-detId|dev.devId)" - same as above
   *    "det-detId|dev.devId" - same as above
   *    "BldInfo(type)" - fully specified BldInfo
   *    "type" - same as above
   *    "BldInfo()" - any BldInfo
   *    "ProcInfo(ipAddr)" - fully specified ProcInfo
   *    "ProcInfo()" - any ProcInfo
   */
  explicit Source(const std::string& spec);

  /**
   *  @brief Assign a string specification.
   */
  Source& operator=(const std::string& spec);

  /**
   *  @brief Returns object which can be used to match Src instances.
   *
   *  If Source instance was constructed from a string then this method tries to
   *  resolve string as an alias. If alias is not found then it tries to parse
   *  the string according to the definitions above. If parsing fails then exception
   *  is thrown.
   *
   *  @param[in] amap Alias map instance.
   *
   *  @throw PSEvt::ExceptionSourceFormat if string cannot be parsed
   */
  SrcMatch srcMatch(const AliasMap& amap) const;

  /// Format Source contents.
  void print(std::ostream& out) const;

protected:

private:

  Pds::Src m_src;
  std::string m_str;

};

/// Helper operator to format Source to a standard stream
inline std::ostream&
operator<<(std::ostream& out, const Source& src)
{
  src.print(out);
  return out;
}

} // namespace PSEvt

#endif // PSEVT_SOURCE_H

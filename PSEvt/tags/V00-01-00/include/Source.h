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

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
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
 *  @brief This class implements source matching for finding things 
 *  inside event.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class Source  {
public:

  enum NoSource { null };
  
  /**
   *  @brief Matches objects without source only.
   */
  Source (NoSource) : m_src() {}

  /**
   *  @brief Match for any source.
   *  
   *  This object will match any source.
   */
  Source ();

  /**
   *  @brief Exact match for source.
   *  
   *  This object will match fully-specified source.
   */
  explicit Source (const Pds::Src& src);
  
  /**
   *  @brief Exact match for DetInfo source.
   *  
   *  This object will match fully-specified DetInfo source.
   */
  Source (Pds::DetInfo::Detector det, uint32_t detId, Pds::DetInfo::Device dev, uint32_t devId);
  
  /**
   *  @brief Exact match for BldInfo.
   *  
   *  This object will match fully-specified BldInfo source.
   */
  explicit Source (Pds::BldInfo::Type type);

  /**
   *  @brief Approximate matching specified via string.
   *  
   *  Format of the match string can be:
   *    "" - match anything
   *    "DetInfo(det.detId:dev.devId)" - fully or partially specified DetInfo
   *    "det.detId:dev.devId" - same as above
   *    "DetInfo(det-detId|dev.devId)" - same as above
   *    "det-detId|dev.devId" - same as above
   *    "BldInfo(type)" - fully or partially specified BldInfo
   *    "type" - same as above
   *    "ProcInfo(ipAddr)" - fully or partially specified ProcInfo
   *  
   *  This is a non-explicit constructor so we allow conversion from std::string. 
   *  
   *  @throw PSEvt::ExceptionSourceFormat if string is not recognized
   */
  Source (const std::string& spec) ;
  
  // Destructor
  ~Source () {}
  
  /**
   *  @brief Assign a string specification.
   *    
   *  @throw PSEvt::ExceptionSourceFormat if string is not recognized
   */
  Source& operator=(const std::string& spec) ;

  /**
   *  @brief Match source with Pds::Src object.
   */
  bool match(const Pds::Src& src) const;
  
  /// Returns true if matches no-source only
  bool isNoSource() const { return m_src == Pds::Src(); }
  
  /// Returns true if it is exact match
  bool isExact() const;
  
  /// Returns the source
  const Pds::Src& src() const { return m_src; }
  
protected:

private:

  // Data members
  Pds::Src m_src;
  
};

} // namespace PSEvt

#endif // PSEVT_SOURCE_H

#ifndef PSEVT_EVENTKEY_H
#define PSEVT_EVENTKEY_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EventKey.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <iosfwd>
#include <typeinfo>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/xtc/Src.hh"
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
 *  @brief Class describing an address or key of the data object in event.
 *  
 *  Event key consists of three components - object type represented by 
 *  its typeinfo pointer, data source address, and string key. 
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see Event
 *
 *  @version \$Id$
 *
 *  @author Andy Salnikov
 */

class EventKey {
public:

  /**
   *  @brief Constructor for EventKey.
   *  
   *  @param[in] typeinfo    Pointer to typeinfo object
   *  @param[in] src         Data source address
   *  @param[in] key         String key
   */
  EventKey (const std::type_info* typeinfo, const Pds::Src& src, const std::string& key)
    : m_typeinfo(typeinfo), m_src(src), m_key(key) 
  {}

  // Destructor
  ~EventKey () {}

  /// Returns special source address which is used for no-source data
  static Pds::Src noSource() { return Pds::Src(); }

  /// Returns special source address which is used for proxies that can serve
  /// any source address
  static Pds::Src anySource() { return Pds::ProcInfo(Pds::Level::NumberOfLevels, 0, 0); }

  /// Compare two keys
  bool operator<(const EventKey& other) const;

  /// Rormat the key
  void print(std::ostream& str) const;

  /// Returns pointer to typeinfo object
  const std::type_info* typeinfo() const {return m_typeinfo;}
  
  /// Returns data source address
  const Pds::Src& src() const {return m_src;}
  
  /// Returns string key
  const std::string& key() const {return m_key;}
  
  /// Returns true if data source address is a valid address.
  bool validSrc() const { return not (m_src == Pds::Src()); }
  
protected:

private:

  // Data members
  const std::type_info* m_typeinfo; ///< Pointer to typeinfo object
  Pds::Src m_src;             ///< Data source address
  std::string m_key;          ///< String key

};

inline
std::ostream&
operator<<(std::ostream& out, const EventKey& key) {
  key.print(out);
  return out;
}

} // namespace PSEvt

/// Helper operator to format Pds::Src to a standard stream
std::ostream&
operator<<(std::ostream& out, const Pds::Src& src);

#endif // PSEVT_EVENTKEY_H

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

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSEvt {

/**
 *  Class describing an addre5ss of the data object in event.
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

class EventKey  {
public:

  // Default constructor
  EventKey (const std::type_info* typeinfo, const Pds::Src& src, const std::string& key)
    : m_typeinfo(typeinfo), m_src(src), m_key(key) 
  {}

  // Destructor
  ~EventKey () {}

  // Compare two keys
  bool operator<(const EventKey& other) const;

  // format the key
  void print(std::ostream& str) const;

  // return typeinfo
  const std::type_info* typeinfo() const {return m_typeinfo;}
  
  // return source
  const Pds::Src& src() const {return m_src;}
  
  // return string key
  const std::string& key() const {return m_key;}
  
  // is src valid
  bool validSrc() const { return not (m_src == Pds::Src()); }
  
protected:

private:

  // Data members
  const std::type_info* m_typeinfo;
  const Pds::Src m_src;
  const std::string m_key;

};

inline
std::ostream&
operator<<(std::ostream& out, const EventKey& key) {
  key.print(out);
  return out;
}

} // namespace PSEvt

#endif // PSEVT_EVENTKEY_H

#ifndef O2OTRANSLATOR_CONFIGOBJECTSTORE_H
#define O2OTRANSLATOR_CONFIGOBJECTSTORE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigObjectStore.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <map>
#include <vector>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/xtc/Src.hh"
#include "pdsdata/xtc/TypeId.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

/**
 *  Class which stores all configuration objects.
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

class ConfigObjectStore {
public:

  typedef std::pair<Pds::TypeId, Pds::Src> key_type;
  // comparison operator for map keys
  struct key_compare {
    bool operator()(const key_type& lhs, const key_type& rhs) const;
  };
  typedef std::multimap<key_type, std::vector<char>, key_compare> ConfigMap;
  typedef ConfigMap::value_type value_type;
  typedef ConfigMap::const_iterator const_iterator;
  typedef ConfigMap::iterator iterator;
  

  // Default constructor
  ConfigObjectStore () ;

  // Destructor
  ~ConfigObjectStore () ;

  // store new config object
  void store(const Pds::TypeId& typeId, const Pds::Src& src, const std::vector<char>& data);

  // find stored config object, return nullptr if not found
  template <typename T>
  const T* find(const Pds::TypeId& typeId, const Pds::Src& src) const {
    return static_cast<const T*>(_find(typeId, src));
  }

  // returns iterators for begin/end of object set
  const_iterator begin() const { return m_config.begin(); }
  const_iterator end() const { return m_config.end(); }

  // reset all contents
  void clear() { return m_config.clear(); }
  
protected:

  // find existing config object, return nullptr if not there
  const void* _find(const Pds::TypeId& typeId, const Pds::Src& src) const;

private:


  // Data members
  ConfigMap m_config ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_CONFIGOBJECTSTORE_H

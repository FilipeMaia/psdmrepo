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

class ConfigObjectStore  {
public:

  // Default constructor
  ConfigObjectStore () ;

  // Destructor
  ~ConfigObjectStore () ;

  // store new config object
  void store(const Pds::TypeId& typeId, const Pds::Src& src, const void* data, uint32_t size);

  // find stored config object
  template <typename T>
  const T* find(const Pds::TypeId& typeId, const Pds::Src& src) const {
    return static_cast<const T*>(_find(typeId, src));
  }

protected:

  // find new config object
  const void* _find(const Pds::TypeId& typeId, const Pds::Src& src) const;

private:

  typedef std::pair<Pds::TypeId, Pds::Src> ConfigKey;
  // comparison operator for map keys
  struct _KeyCmp {
    bool operator()( const ConfigKey& lhs, const ConfigKey& rhs ) const ;
  };
  typedef std::map<ConfigKey, const void*, _KeyCmp> ConfigMap ;

  // Data members
  ConfigMap m_config ;

  // Copy constructor and assignment are disabled by default
  ConfigObjectStore ( const ConfigObjectStore& ) ;
  ConfigObjectStore& operator = ( const ConfigObjectStore& ) ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_CONFIGOBJECTSTORE_H

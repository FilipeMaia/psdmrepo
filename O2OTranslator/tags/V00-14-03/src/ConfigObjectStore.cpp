//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigObjectStore...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/ConfigObjectStore.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {


// comparison operator for Src objects
bool
ConfigObjectStore::_KeyCmp::operator()( const ConfigKey& lhs, const ConfigKey& rhs ) const
{
  if ( lhs.first.value() < rhs.first.value() ) return true ;
  if ( lhs.first.value() > rhs.first.value() ) return false ;
  if ( lhs.second.log() < rhs.second.log() ) return true ;
  if ( lhs.second.log() > rhs.second.log() ) return false ;
  if ( lhs.second.phy() < rhs.second.phy() ) return true ;
  return false ;
}


//----------------
// Constructors --
//----------------
ConfigObjectStore::ConfigObjectStore ()
  : m_config()
{
}

//--------------
// Destructor --
//--------------
ConfigObjectStore::~ConfigObjectStore ()
{
  for ( ConfigMap::iterator it = m_config.begin() ; it != m_config.end() ; ++ it ) {
    delete [] (char*)it->second;
  }
}

// store new config object
void
ConfigObjectStore::store(const Pds::TypeId& typeId, const Pds::Src& src, const void* data, uint32_t size)
{
  ConfigKey key(typeId, src);
  ConfigMap::iterator it = m_config.find(key);
  if ( it != m_config.end() ) {
    // delete old object, replace with new
    char* newobj = new char[size];
    std::copy( (const char*)data, ((const char*)data)+size, newobj);
    delete [] (char*)it->second;
    it->second = newobj;
  } else {
    // create new one
    char* newobj = new char[size];
    std::copy( (const char*)data, ((const char*)data)+size, newobj);
    m_config.insert(ConfigMap::value_type(key, newobj));
  }
}

// find new config object
const void*
ConfigObjectStore::_find(const Pds::TypeId& typeId, const Pds::Src& src) const
{
  ConfigKey key(typeId, src);
  ConfigMap::const_iterator it = m_config.find(key);
  if ( it != m_config.end() ) return it->second;
  return 0;
}

} // namespace O2OTranslator

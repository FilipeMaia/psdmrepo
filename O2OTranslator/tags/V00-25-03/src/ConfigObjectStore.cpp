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
ConfigObjectStore::key_compare::operator()( const key_type& lhs, const key_type& rhs ) const
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
}

// store new config object
void
ConfigObjectStore::store(const Pds::TypeId& typeId, const Pds::Src& src, const std::vector<char>& data)
{
  key_type key(typeId, src);
  m_config.insert(value_type(key, data));
}

// find config object
const void*
ConfigObjectStore::_find(const Pds::TypeId& typeId, const Pds::Src& src) const
{
  key_type key(typeId, src);
  const_iterator it = m_config.find(key);
  if ( it != m_config.end() ) return it->second.data();
  return 0;
}

} // namespace O2OTranslator

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ProxyDict...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSEvt/ProxyDict.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSEvt/Exceptions.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSEvt {

//----------------
// Constructors --
//----------------
ProxyDict::ProxyDict ()
{
}

//--------------
// Destructor --
//--------------
ProxyDict::~ProxyDict ()
{
}


void 
ProxyDict::putImpl( const boost::shared_ptr<ProxyI>& proxy, 
                    const std::type_info* typeinfo, 
                    const Pds::Src& source, 
                    const std::string& key )
{
  EventKey proxyKey(typeinfo, source, key);
  
  // there should not be existing key
  Dict::iterator it = m_dict.find(proxyKey);
  if ( it != m_dict.end() ) {
    throw ExceptionDuplicateKey(ERR_LOC, typeinfo, source, key);
  }

  m_dict.insert(Dict::value_type(proxyKey, proxy));
}


boost::shared_ptr<void> 
ProxyDict::getImpl( const std::type_info* typeinfo, 
                    const Pds::Src& source, 
                    const std::string& key )
{
  EventKey proxyKey(typeinfo, source, key);

  Dict::const_iterator it = m_dict.find(proxyKey);
  if ( it != m_dict.end() ) {
    // call proxy to get the value
    return it->second->get(this, source, key);
  }
  return proxy_ptr();
}

bool 
ProxyDict::existsImpl( const std::type_info* typeinfo, 
                       const Pds::Src& source, 
                       const std::string& key)
{
  EventKey proxyKey(typeinfo, source, key);
  return m_dict.find(proxyKey) != m_dict.end();
}

bool 
ProxyDict::removeImpl( const std::type_info* typeinfo, 
                       const Pds::Src& source, 
                       const std::string& key )
{
  EventKey proxyKey(typeinfo, source, key);
  return m_dict.erase(proxyKey) > 0;
}

void 
ProxyDict::keysImpl(std::list<EventKey>& keys) const
{
  keys.clear();
  for (Dict::const_iterator it = m_dict.begin(); it != m_dict.end(); ++ it) {
    keys.push_back(it->first);
  }
}

} // namespace PSEvt

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
                    const Pds::DetInfo& detInfo, 
                    const std::string& key )
{
  Key proxyKey(typeinfo, detInfo, key);
  
  // there should not be existing key
  Dict::iterator it = m_dict.find(proxyKey);
  if ( it != m_dict.end() ) {
    throw ExceptionDuplicateKey(ERR_LOC, typeinfo, detInfo, key);
  }

  m_dict.insert(Dict::value_type(proxyKey, proxy));
}


boost::shared_ptr<void> 
ProxyDict::getImpl( const std::type_info* typeinfo, 
                    const Pds::DetInfo& detInfo, 
                    const std::string& key )
{
  Key proxyKey(typeinfo, detInfo, key);

  Dict::const_iterator it = m_dict.find(proxyKey);
  if ( it != m_dict.end() ) {
    // call proxy to get the value
    return it->second->get(this, detInfo, key);
  }
  return proxy_ptr();
}

bool 
ProxyDict::existsImpl( const std::type_info* typeinfo, 
                       const Pds::DetInfo& detInfo, 
                       const std::string& key)
{
  Key proxyKey(typeinfo, detInfo, key);
  return m_dict.find(proxyKey) != m_dict.end();
}

bool 
ProxyDict::removeImpl( const std::type_info* typeinfo, 
                       const Pds::DetInfo& detInfo, 
                       const std::string& key )
{
  Key proxyKey(typeinfo, detInfo, key);
  return m_dict.erase(proxyKey) > 0;
}

} // namespace PSEvt

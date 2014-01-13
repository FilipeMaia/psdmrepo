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
ProxyDict::ProxyDict (const boost::shared_ptr<AliasMap>& amap)
  : m_dict()
  , m_amap(amap)
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
                    const EventKey& key )
{
  // key may define one of alias or Src
  EventKey ekey = key;
  if (not ekey.alias().empty()) {
    if (m_amap) {
      // get src from alias
      Pds::Src src = m_amap->src(ekey.alias());
      ekey = EventKey(ekey.typeinfo(), src, ekey.key(), ekey.alias());
    } else {
      throw ExceptionNoAliasMap(ERR_LOC);
    }
  } else {
    if (m_amap) {
      // try to find alias for src
      const std::string& alias = m_amap->alias(ekey.src());
      ekey = EventKey(ekey.typeinfo(), ekey.src(), ekey.key(), alias);
    }
  }

  // there should not be existing key
  Dict::iterator it = m_dict.find(ekey);
  if ( it != m_dict.end() ) {
    throw ExceptionDuplicateKey(ERR_LOC, ekey);
  }

  m_dict.insert(Dict::value_type(ekey, proxy));
}


boost::shared_ptr<void> 
ProxyDict::getImpl( const std::type_info* typeinfo, 
                    const Source& source, 
                    const std::string& key,
                    Pds::Src* foundSrc )
{
  Source::SrcMatch srcm = source.srcMatch(m_amap ? *m_amap : AliasMap());

  if (srcm.isExact()) {
    
    EventKey proxyKey(typeinfo, srcm.src(), key);
    Dict::const_iterator it = m_dict.find(proxyKey);
    if ( it != m_dict.end() ) {
      // call proxy to get the value
      if (foundSrc) *foundSrc = it->first.src();
      return it->second->get(this, it->first.src(), key);
    }

    // try special any-source proxy
    EventKey proxyKeyAny(typeinfo, EventKey::anySource(), key);
    it = m_dict.find(proxyKeyAny);
    if ( it != m_dict.end() ) {
      // call proxy to get the value
      if (foundSrc) *foundSrc = srcm.src();
      return it->second->get(this, srcm.src(), key);
    }

  } else {

    // When source is a match then no-source objects have priority. Try to
    // find no-source object first and see if it matches
    if (srcm.match(Pds::Src())) {
      EventKey proxyKey(typeinfo, Pds::Src(), key);
      Dict::const_iterator it = m_dict.find(proxyKey);
      if ( it != m_dict.end() ) {
        // call proxy to get the value
        if (foundSrc) *foundSrc = it->first.src();
        return it->second->get(this, it->first.src(), key);
      }
    }

    // Do linear search, find first match
    for (Dict::const_iterator it = m_dict.begin(); it != m_dict.end(); ++ it) {
      if (*typeinfo == *it->first.typeinfo() and
          key == it->first.key() and 
          srcm.match(it->first.src()) ) {
        // call proxy to get the value
        if (foundSrc) *foundSrc = it->first.src();
        return it->second->get(this, it->first.src(), key);
      }
    }
    
  }

  return proxy_ptr();

}

bool 
ProxyDict::existsImpl(const EventKey& key)
{
  return m_dict.find(key) != m_dict.end();
}

bool 
ProxyDict::removeImpl(const EventKey& key)
{
  return m_dict.erase(key) > 0;
}

void 
ProxyDict::keysImpl(std::list<EventKey>& keys, const Source& source) const
{
  Source::SrcMatch srcm = source.srcMatch(m_amap ? *m_amap : AliasMap());

  keys.clear();
  for (Dict::const_iterator it = m_dict.begin(); it != m_dict.end(); ++ it) {
    if (srcm.match(it->first.src())) keys.push_back(it->first);
  }
}

} // namespace PSEvt
